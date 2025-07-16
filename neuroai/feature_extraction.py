import os
import glob
import h5py
import pandas as pd
import numpy as np
import torch
import math
import librosa
import speechbrain
import nibabel as nib
from speechbrain.inference import EncoderClassifier, SpeakerRecognition
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from PIL import Image # For image processing (conceptual)
from torchvision import transforms # For image processing (conceptual)
from transformers import AutoProcessor, CLIPModel

# --- Global Model/Processor Caches (to avoid reloading for every call) ---
_MODEL_CACHE = {}

def _get_model_and_processor(model_id: str, checkpoint: str = None):
    """
    Loads and caches the specified AI model and its processor.
    """
    cache_key = f"{model_id}_{checkpoint}"
    if cache_key in _MODEL_CACHE:
        print("here")
        return _MODEL_CACHE[cache_key]

    print(f"Loading {model_id} model and processor (checkpoint: {checkpoint})...")
    model = None
    processor = None

    if model_id.lower() == 'ecapa-tdnn':
        if checkpoint and checkpoint != "speechbrain/spkrec-ecapa-voxceleb":
            print(f"Warning: ECAPA-TDNN only supports 'speechbrain/spkrec-ecapa-voxceleb' checkpoint. Ignoring '{checkpoint}'.")
        model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.join("tmp", "ecapa_model_cache"))
    elif model_id.lower().startswith('whisper'):
        if checkpoint:
            model_name = checkpoint
        else:
            raise ValueError(
            "You must specify a valid checkpoint name for the pretrained Whisper model.\n"
            "Examples include: 'openai/whisper-tiny', 'openai/whisper-base'.\n"
            "See the full list of available checkpoints at: https://huggingface.co/openai"
        )
        try:
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
        except Exception as e:
             raise ValueError(
            f"Failed to load the pretrained Whisper model from checkpoint '{model_name}'.\n"
            f"Please ensure the checkpoint name is correct and available on Hugging Face.\n"
            f"Original error: {str(e)}"
    )
    elif model_id.lower() == 'clip':
        if checkpoint:
            model_name = checkpoint
        else:
            raise ValueError(
            "You must specify a valid checkpoint name for the pretrained Whisper model.\n"
            "Examples include: 'openai/clip-vit-large-patch14', 'openai/clip-vit-base-patch32'.\n"
            "See the full list of available checkpoints at: https://huggingface.co/openai"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    elif model_id.lower() == 'resnet':
        from torchvision.models import resnet50, ResNet50_Weights
        if checkpoint and checkpoint != "IMAGENET1K_V1":
             print(f"Warning: ResNet50 only supports 'IMAGENET1K_V1' weights by default. Ignoring '{checkpoint}'.")
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        model.eval() # Set to evaluation mode
        processor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

    # Move model to GPU if available
    if torch.cuda.is_available() and hasattr(model, 'to'):
        model.to('cuda')
    
    _MODEL_CACHE[cache_key] = (model, processor)
    return model, processor

# --- Model-specific feature extraction functions ---

def _extract_ecapa_embeddings(audio_segment: np.ndarray, model: EncoderClassifier, layer_name: str = 'final') -> np.ndarray:
    """
    Extracts embeddings from ECAPA-TDNN for a given audio segment and layer.
    """
    signal = torch.tensor(audio_segment).unsqueeze(0)
    if torch.cuda.is_available():
        signal = signal.to('cuda')

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        if layer_name == 'final':
            # This corresponds to the `embedding` in your original script
            embeddings = model.encode_batch(signal).squeeze().cpu().numpy()
            return embeddings
        if layer_name.lower() not in ["TDNN".lower(), "SERes2Net1".lower(), "SERes2Net2".lower(),"SERes2Net3".lower(),"MFA".lower(), "AttentiveStatisticsPooling".lower(), "AttentiveStatisticsPooling-BatchNorm1d".lower(), "FC".lower()]:
            raise ValueError(f"ECAPA-TDNN layer '{layer_name}' not found. Available block layers are TDNN, SERes2Net1, SERes2Net2, SERes2Net3, MFA, AttentiveStatisticsPooling, AttentiveStatisticsPooling-BatchNorm1d and FC.")
        
        # For intermediate layers
        features = model.mods.compute_features(signal)
        signal_length = torch.tensor([features.shape[1]], dtype=torch.long, device=signal.device)
        features = model.mods.mean_var_norm(features, signal_length)

        x = features.transpose(1, 2)
        
        block_outputs = []  
        for idx, layer in enumerate(model.mods.embedding_model.blocks):
            x = layer(x)
            if layer_name.lower() == "tdnn" and idx==0:
                return x[0].transpose(0,1).cpu().numpy()
            if layer_name.lower().startswith("seres2net"):
                block_idx = int(layer_name.replace("seres2net", ""))
                if idx == block_idx:
                    return x[0].transpose(0,1).cpu().numpy()
            if isinstance(layer, speechbrain.lobes.models.ECAPA_TDNN.SERes2NetBlock):
                block_outputs.append(x)

        aggregated_features = torch.cat(block_outputs, dim=1)  
        mfa_output = model.mods.embedding_model.mfa(aggregated_features)
        if layer_name.lower() == "mfa":
            return mfa_output[0].transpose(0,1).cpu().numpy()
        pooled = model.mods.embedding_model.asp(mfa_output)
        if layer_name.lower() == "attentivestatisticspooling":
            return np.squeeze(np.array(pooled)).cpu().numpy()
        
        pooled = model.mods.embedding_model.asp_bn(pooled)
        if layer_name.lower() == "attentivestatisticspooling-batchnorm1d":
            return np.squeeze(np.array(pooled)).cpu().numpy()
        embedding = model.mods.embedding_model.fc(pooled)
        return np.squeeze(np.array(embedding))


def _extract_whisper_embeddings(audio_segment: np.ndarray, model: WhisperForConditionalGeneration, processor: WhisperProcessor, layer_name: str = 'final') -> np.ndarray:
    """
    Extracts embeddings from Whisper model for a given audio segment and layer.
    Handles 'encoder_layerX' and 'decoder_layerX'.
    """
    audio_input = processor(
        audio_segment,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_timestamps= True,
        
    )
    input_features= audio_input['input_features']
    num_frames= audio_input["num_frames"]
    idx= math.ceil(num_frames/2)
    if torch.cuda.is_available():
        input_features = input_features.to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            input_features, output_hidden_states=True, return_dict_in_generate=True, use_cache=True, attention_mask= audio_input['attention_mask'])
        if layer_name == 'final':  
            embedding_with_padding= outputs.encoder_hidden_states[-1][0].cpu().numpy()
            return embedding_with_padding[:idx] # (sequence_length, hidden_size)
        
        elif layer_name.startswith('encoder_layer'):
            layer_idx = int(layer_name.replace('encoder_layer', ''))
            if layer_idx < len(outputs.encoder_hidden_states):
                embedding_with_padding= outputs.encoder_hidden_states[layer_idx][0].cpu().numpy()
                return embedding_with_padding[:idx] # (sequence_length, hidden_size)
            else:
                raise ValueError(f"Whisper encoder layer '{layer_name}' not found. "
                                 f"Max encoder layer is layer {len(outputs.encoder_hidden_states) - 1}.")
        
        elif layer_name.startswith('decoder_layer'):
            layer_idx = int(layer_name.replace('decoder_layer', ''))

            decoder_hidden_states = outputs.decoder_hidden_states # (num_generated_tokens, num_decoder_layers, batch_size, sequence_length, hidden_size)

            if layer_idx < len(decoder_hidden_states[0]): # Check against number of layers
                # Collect embeddings for the specified layer across all generated tokens
                layer_embeddings = []
                for i in range(len(decoder_hidden_states)): #Tuple (num_tokens, num_layers, batch, sequence_length, embedding_dim)   
                    for j in range(len(decoder_hidden_states[i][layer_idx][0])):
                        layer_embeddings.append(decoder_hidden_states[i][layer_idx][0][j])
                layer_embeddings= np.array(layer_embeddings)[3:-1] # remove embeddings corresponding to the special tokens

                return layer_embeddings
            else:
                raise ValueError(f"Whisper decoder layer '{layer_name}' not found. "
                                 f"Max decoder layer is {len(decoder_hidden_states[0]) - 1}.")
        else:
            raise ValueError(f"Whisper layer name '{layer_name}' not recognized. "
                             "Use 'final encoder layer', 'encoder_layerX', or 'decoder_layerX'.")


# --- Main Feature Extraction Function ---

def extract_features(
    bold_data_root_dir: str,
    stimuli_base_dir: str,
    output_file_path: str,
    model_id: str,
    layer_name: str = 'final',
    checkpoint: str = None,
    tr: float = 2.0, # TR duration in seconds
    task_of_interest: str = None, # instead of processing all files in the stimuli folder, specify the folders or files of interest 
):
    """
    Extracts AI model features for fMRI stimulus events based on _events.tsv files.

    Parameters
    ----------
    bold_data_root_dir: str
        Path to the root of your BIDS dataset. This is used to find the tasks
         `_events.tsv` files.
    stimuli_base_dir : str
        Path to the root directory containing your raw stimulus files
        (e.g., 'data/stimuli/'). Paths in `_events.tsv`'s 'stim_file' column
        should be relative to this directory.
    output_file_path : str
        Path to the H5 file where extracted features will be saved.
        Keys will be `functional_run_name` (e.g.,'coco_run-01').
    model_id : str
        Identifier for the AI model to use (e.g., 'whisper', 'ecapa-tdnn', 'clip', 'resnet').
    layer_name : str, optional
        Specific layer from which to extract features. Defaults to 'final'.
        - For ECAPA-TDNN: 'final', 'TDNN', 'SERes2Net1', 'SERes2Net2','SERes2Net3', 'MFA',
          'AttentiveStatisticsPooling', 'AttentiveStatisticsPooling-BatchNorm1d', 'FC'
        - For Whisper: 'final' (which is the encoder's final layer), 'encoder_layerX', 'decoder_layerX'
        - For CLIP and ResNet: Currently, only the 'final' layer of the model's feature extractor is supported.
    checkpoint : str, 
        Specific checkpoint/version of the model, Required if the selected model has multiple checkpoints (e.g., 'openai/whisper-large').
    tr : float,
        The TR (repetition time) duration of your fMRI data in seconds.
        This is crucial for determining how to segment stimuli into TR-aligned features. Defaults to 2.0.
    task_of_interest : str, optional
        If provided, only `_events.tsv` files corresponding to this specific
        task/run name will be processed.
        If None, all tasks found will be processed.
    """


    model, processor = _get_model_and_processor(model_id, checkpoint)
    
    # Ensure model is on GPU if available and applicable
    if torch.cuda.is_available() and hasattr(model, 'to'):
        model.to('cuda')

    # Find all _events.tsv files in the BIDS dataset
    event_files = glob.glob(os.path.join(bold_data_root_dir, 'sub-*','ses-*', 'func', '*_events.tsv'))
    if not event_files:
        raise FileNotFoundError(f"No *_events.tsv files found in '{bold_data_root_dir:}'. "
                                "Ensure your data follows BIDS structure with event files.")

    # Use 'a' mode for h5py to append to existing file or create if not exists
    with h5py.File(output_file_path, 'a') as hf:
        # Iterate over each event file
        for event_file_path in event_files:
            print(f"\nProcessing events file: {event_file_path}")
            
            # Derive the functional_run_name from the event file path
            filename_parts = os.path.basename(event_file_path).split('_')
            
            # Find the 'task-' part and extract the task name, and potentially run index
            functional_run_name_parts = []
            for part in filename_parts:
                if part.startswith('task-'):
                    functional_run_name_parts.append(part.replace('task-', ''))
                elif part.startswith('run-'): # If 'run-X' directly follows 'task-Y', append it
                    if functional_run_name_parts: # only append run if a task was found
                         functional_run_name_parts.append(part)     
            functional_run_name = "_".join(functional_run_name_parts)

            if not functional_run_name:
                print(f"Could not determine functional run name from {event_file_path}. Skipping.")
                continue

            if task_of_interest and functional_run_name != task_of_interest:
                print(f"Skipping run '{functional_run_name}' as it's not the task of interest ('{task_of_interest}').")
                continue

            # Check if this run's features already exist in the H5 file
            if functional_run_name in hf:
                print(f"Features for '{functional_run_name}' already exist in {output_file_path}. Skipping.")
                continue

            try:
                events_df = pd.read_csv(event_file_path, sep='\t')
                events_df.columns = events_df.columns.str.lower()
                if 'onset' not in events_df.columns or 'duration' not in events_df.columns or 'stim_file' not in events_df.columns:
                    raise ValueError(f"Required columns (onset, duration, stim_file) not found in {event_file_path}")
            except Exception as e:
                print(f"Error reading or parsing {event_file_path}. Skipping. Error: {e}")
                continue
            

            # Determine the total number of TRs for this run from a corresponding BOLD file or from the total duration of the run
            bold_file_path= event_file_path.replace("_events.tsv", "_bold.dtseries.nii")
            matching_bold_files = glob.glob(bold_file_path)

            if not matching_bold_files:
                print(f"Warning: No BOLD file found for functional run '{functional_run_name}'. "
                      "The number of TRs will be estimated from the duration information in the events.tsv file.")
            
            if matching_bold_files:
                # Load the first matching BOLD file to get its number of TRs
                try:
                    first_bold_nifti = nib.load(matching_bold_files[0])
                    num_TRs = first_bold_nifti.shape[0]
                    print(f"Identified {num_TRs} TRs for run '{functional_run_name}' from BOLD data.")
                except Exception as e:
                    print(f"Error loading BOLD file '{matching_bold_files[0]}' to get TR count. Skipping. Error: {e}")
                    continue
            else:
                total_duration = events_df['onset'].values[-1] + events_df['duration'].values[-1]
                num_TRs = int(np.floor(total_duration / tr))
            feature_matrix = []
            feature_dim = None
            for tr_idx in range(num_TRs):
                tr_start = tr_idx * tr
                tr_end = tr_start + tr
                tr_features = []

                for _, row in events_df.iterrows():
                    onset, duration, stim_file = row['onset'], row['duration'], row['stim_file']
                    stim_file_path = os.path.join(stimuli_base_dir, stim_file)
                    stim_start = onset
                    stim_end = onset + duration

                    # Check if stimulus overlaps with current TR
                    overlap_start = max(tr_start, stim_start)
                    overlap_end = min(tr_end, stim_end)
                    if overlap_start < overlap_end:
                        if stim_file.lower().endswith(('.wav', '.mp3', '.flac')):
                            # Load and resample audio
                            raw_audio, original_sr = librosa.load(stim_file_path, sr=None)
                            if original_sr != 16000:
                                audio_input = librosa.resample(raw_audio, orig_sr=original_sr, target_sr=16000)
                            else:
                                audio_input = raw_audio

                            # Trim audio to the overlapping segment
                            segment_start = overlap_start - stim_start
                            segment_end = overlap_end - stim_start
                            start_sample = int(segment_start * 16000)
                            end_sample = int(segment_end * 16000)
                            audio_segment = audio_input[start_sample:end_sample]
                            if model_id.lower() == 'ecapa-tdnn':
                                # _extract_ecapa_embeddings returns (sequence_length, feature_dim) or (feature_dim,)
                                segment_features = _extract_ecapa_embeddings(audio_segment, model, layer_name)
                            
                            elif model_id.lower().startswith('whisper'):

                                segment_features = _extract_whisper_embeddings(audio_segment, model, processor, layer_name)

                            # Add other audio models here
                            else:
                                raise ValueError(f"Audio model '{model_id}' not yet implemented or misconfigured.")

                            if segment_features is not None and len(segment_features) > 0:
                                tr_features.append(segment_features)

                        elif stim_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                            # --- Placeholder for Image Models (e.g., CLIP, ResNet) ---
                            # You'd add your image loading and processing logic here
                            # Example for ResNet:
                            try:
                                image = Image.open(stim_file_path).convert('RGB')
                                if torch.cuda.is_available():
                                    image_input = image_input.to('cuda')
                                with torch.no_grad():
                                    if model_id.lower() == 'resnet':
                                        if layer_name == 'final':
                                            image_input = processor(image).unsqueeze(0)
                                            # Get features before final classification layer
                                            features = model(image_input).squeeze().cpu().numpy()
                                            print(features.shape)
                                        else:
                                            raise ValueError(f"ResNet layer '{layer_name}' not supported yet only final layer is provided.")
                                    elif model_id.lower() == 'clip':
                                        if layer_name == 'final':
                                            inputs = processor(images=image, return_tensors="pt")
                                            # CLIP's image features
                                            features = model.get_image_features(**inputs)
                                            print(features.shape)
                                        else:
                                            raise ValueError(f"clip layer '{layer_name}' not supported yet only final layer is provided.")
                                    else:
                                        raise ValueError(f"Image model '{model_id}' not yet implemented or misconfigured.")
                                if not len(features.shape) > 1:
                                    segment_features = features[np.newaxis, :] # Make it (1, feature_dim) for a single image
                                else:
                                    segment_features = features
                                tr_features.append(segment_features)
                            except Exception as e:
                                print(f"Error processing image file {stim_file_path}: {e}. Skipping event.")
                                continue
                            print(f"Warning: Image processing for '{model_id}' is a placeholder. Skipping image event.")
                            continue
                        else:
                            print(f"Warning: Unsupported stimulus file type: {stim_file_path}. Skipping event.")
                            continue
                # Average features for current TR
                if tr_features:
                    tr_features_concat = np.concatenate(tr_features, axis=0)  # [N, feature_dim]
                    if len(np.array(tr_features_concat).shape) > 1:
                        print(np.array(tr_features).shape)
                        tr_features_mean = torch.from_numpy(tr_features_concat).mean(dim=0)    # [feature_dim]
                        tr_features_mean= np.array(tr_features_mean)
                    else:
                        tr_features_mean= tr_features_concat   
                        print("here", tr_features_mean.shape)    
                else:
                    # Default to zeros if no features (use known or fallback dim)
                    tr_features_mean = np.zeros(feature_dim) if feature_dim else np.zeros(768)

                # Initialize feature_dim on first non-zero entry
                if feature_dim is None and tr_features:
                    feature_dim = tr_features_mean.shape[0]

                feature_matrix.append(tr_features_mean)
            
            feature_matrix= np.stack(feature_matrix)
                
            print(f"Saving features for run '{functional_run_name}' to {output_file_path}...")
            task= hf.create_group(functional_run_name)
            task.create_dataset("feature_matrix", data=feature_matrix)
            print(f"Successfully processed and saved features for '{functional_run_name}' with shape {feature_matrix.shape}.")
        else:
            print(f"No features extracted for functional run '{functional_run_name}'. Not saving.")

    print(f"\nFeature extraction complete. Output saved to: {output_file_path}")
    
extract_features(bold_data_root_dir= r"E:\year5\GP\padded_stimuli\New folder", stimuli_base_dir= r"E:\year5\GP\padded_stimuli", output_file_path= r"E:\year5\GP\padded_stimuli\new\whisper_decoder_layer6.h5", model_id= "whisper", layer_name="decoder_layer6", checkpoint="openai/whisper-base", tr= 2, task_of_interest= "adollshouse")