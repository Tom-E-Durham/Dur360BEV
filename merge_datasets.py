import os
import shutil
import pickle
import sys # Added for sys.exit()
import argparse # Added for command-line arguments
import numpy as np # Added for numpy array concatenation

# --- Argument Parsing --- 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge two Dur360BEV-like datasets. " \
                                                 "Modifies the initial dataset in-place and then renames it.")
    parser.add_argument("--folder_dir", 
                        type=str, 
                        required=True, 
                        help="The main base path for all dataset operations. " \
                              "This directory should contain the initial and extended dataset subfolders.")
    
    # Arguments for dataset structure names with defaults
    parser.add_argument("--initial_subfolder", type=str, default="extract_initial",
                        help="Name of the subfolder containing the initial dataset (default: 'extract_initial').")
    parser.add_argument("--initial_dataset_name", type=str, default="Dur360BEV_Dataset",
                        help="Name of the initial dataset directory (default: 'Dur360BEV_Dataset').")
    parser.add_argument("--extended_subfolder", type=str, default="extract_extended",
                        help="Name of the subfolder containing the extended dataset (default: 'extract_extended').")
    parser.add_argument("--extended_dataset_name", type=str, default="Dur360BEV_Dataset_Extended",
                        help="Name of the extended dataset directory (default: 'Dur360BEV_Dataset_Extended').")
    parser.add_argument("--final_merged_name", type=str, default="Dur360BEV_Dataset_Complete",
                        help="Name for the final merged dataset directory (default: 'Dur360BEV_Dataset_Complete').")
    
    args = parser.parse_args()
    return args

# --- Main Script Logic --- 
if __name__ == "__main__":
    args = parse_arguments()
    main_datasets_folder_path = args.folder_dir

    if not os.path.isdir(main_datasets_folder_path):
        print(f"Error: The provided folder_dir '{main_datasets_folder_path}' is not a valid directory.")
        sys.exit(1)

    print(f"Using main datasets folder path: {main_datasets_folder_path}")

    # Define specific dataset paths based on arguments
    initial_dataset_subfolder = args.initial_subfolder
    initial_dataset_name = args.initial_dataset_name
    extended_dataset_subfolder = args.extended_subfolder
    extended_dataset_name = args.extended_dataset_name
    final_merged_dataset_name = args.final_merged_name

    initial_dataset_path = os.path.join(main_datasets_folder_path, initial_dataset_subfolder, initial_dataset_name)
    extended_dataset_path = os.path.join(main_datasets_folder_path, extended_dataset_subfolder, extended_dataset_name)
    # Ensure the parent directory for the final merged dataset exists before trying to get its dirname for the target path.
    # This is relevant if initial_dataset_subfolder is empty or points to the root of main_datasets_folder_path.
    # However, os.path.dirname on a path like "Dur360BEV_Dataset_Complete" (if initial_dataset_path was just its name)
    # would return an empty string, so joining it with main_datasets_folder_path is safer.
    
    # Determine the directory where the final merged dataset will reside.
    # This is the directory containing the initial_dataset_path.
    final_merged_dataset_parent_dir = os.path.dirname(initial_dataset_path)
    final_merged_dataset_target_path = os.path.join(final_merged_dataset_parent_dir, final_merged_dataset_name)


    # Check if the constructed dataset paths exist
    if not os.path.isdir(initial_dataset_path):
        print(f"Error: The initial dataset path does not exist or is not a directory: {initial_dataset_path}")
        print(f"       Constructed from --folder_dir='{main_datasets_folder_path}', --initial_subfolder='{initial_dataset_subfolder}', --initial_dataset_name='{initial_dataset_name}'")
        sys.exit(1)
    
    if not os.path.isdir(extended_dataset_path):
        print(f"Error: The extended dataset path does not exist or is not a directory: {extended_dataset_path}")
        print(f"       Constructed from --folder_dir='{main_datasets_folder_path}', --extended_subfolder='{extended_dataset_subfolder}', --extended_dataset_name='{extended_dataset_name}'")
        sys.exit(1)

    print(f"Initial dataset path: {initial_dataset_path}")
    print(f"Extended dataset path: {extended_dataset_path}")
    print(f"Final merged dataset target path: {final_merged_dataset_target_path}")
    
    # Data types to process (still hardcoded as it defines the expected content types)
    data_types = ["image", "labels", "ouster_points", "oxts"]

    def count_lines_in_file(filepath):
        """Counts lines in a file. Returns 0 if file not found or empty."""
        if not os.path.exists(filepath):
            return 0
        try:
            with open(filepath, 'r') as f:
                lines = sum(1 for line in f)
            return lines
        except Exception as e:
            print(f"Could not read lines from {filepath}: {e}")
            return -1 # Indicate an error in reading

    def check_dataset_integrity(dataset_path, dataset_name_for_error_msg, d_types):
        """
        Checks the integrity of a dataset.
        - For each data type, number of data files should match timestamp lines.
        - All data types should have the same length.
        Returns a list of error messages. Empty if no errors.
        """
        errors = []
        type_lengths = {} # To store length for each data type

        print(f"--- Checking integrity of {dataset_name_for_error_msg} at {dataset_path} ---")

        for data_type in d_types:
            data_dir = os.path.join(dataset_path, data_type, "data")
            timestamps_file = os.path.join(dataset_path, data_type, "timestamps.txt")

            num_data_files = 0
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                num_data_files = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
            else:
                # If data_dir doesn't exist, it's an issue unless timestamps also don't exist or are empty
                if os.path.exists(timestamps_file) and count_lines_in_file(timestamps_file) > 0:
                     errors.append(f"[{dataset_name_for_error_msg} - {data_type}]: Data directory '{data_dir}' missing, but timestamps file '{timestamps_file}' exists and is not empty.")
                # If both are missing/empty, it's consistent for this type in isolation, but might fail cross-type check

            num_timestamp_lines = count_lines_in_file(timestamps_file)
            if num_timestamp_lines == -1: # Error reading timestamp file
                errors.append(f"[{dataset_name_for_error_msg} - {data_type}]: Could not read timestamps file '{timestamps_file}'.")
                type_lengths[data_type] = -1 # Mark as error for cross-check
                continue


            if num_data_files != num_timestamp_lines:
                errors.append(f"[{dataset_name_for_error_msg} - {data_type}]: Mismatch! Data files: {num_data_files}, Timestamp lines: {num_timestamp_lines}.")
            
            type_lengths[data_type] = num_data_files # Use data_files count as the reference length for this type

            print(f"  {data_type}: {num_data_files} data files, {num_timestamp_lines} timestamp lines.")


        # Check for consistency across data types
        if not type_lengths: # Should not happen if d_types is not empty
            errors.append(f"[{dataset_name_for_error_msg}]: No data types processed or found to compare lengths.")
            return errors

        # Filter out types that had reading errors for the cross-type length check
        valid_lengths = {dt: length for dt, length in type_lengths.items() if length != -1}

        if valid_lengths:
            first_type_name = list(valid_lengths.keys())[0]
            reference_length = valid_lengths[first_type_name]
            
            for data_type, length in valid_lengths.items():
                if length != reference_length:
                    errors.append(f"[{dataset_name_for_error_msg}]: Inconsistent lengths across data types. '{first_type_name}' has {reference_length} items, but '{data_type}' has {length} items.")
                    # No need to report for all pairs, one such message is enough to indicate cross-type inconsistency.
                    break 
        elif errors: # Only errors, no valid lengths to compare
            pass # Errors already logged
        else: # No valid lengths and no errors means all types might have had -1 length (e.g. all timestamps unreadable)
            errors.append(f"[{dataset_name_for_error_msg}]: Could not determine a reference length for cross-type comparison due to issues with all data types.")


        if not errors:
            print(f"--- Integrity check PASSED for {dataset_name_for_error_msg} ---")
        else:
            print(f"--- Integrity check FAILED for {dataset_name_for_error_msg} ---")
        return errors

    # --- Perform Integrity Checks Before Starting Merge ---
    all_errors = []
    print("\nPerforming pre-merge integrity checks...")

    # Check initial dataset
    initial_errors = check_dataset_integrity(initial_dataset_path, f"Initial Dataset ({initial_dataset_name})", data_types)
    all_errors.extend(initial_errors)

    # Check extended dataset
    extended_errors = check_dataset_integrity(extended_dataset_path, f"Extended Dataset ({extended_dataset_name})", data_types)
    all_errors.extend(extended_errors)

    if all_errors:
        print("\n---------------------------------------------------------")
        print("ERROR: Dataset integrity checks failed. Please resolve the following issues:")
        for error_msg in all_errors:
            print(f"- {error_msg}")
        print("Merging process will be skipped due to integrity issues.")
        print("---------------------------------------------------------")
        sys.exit(1) # Exit the script
    else:
        print("\nAll dataset integrity checks passed. Proceeding with merge.")
        print("---------------------------------------------------------")


    # No initial full copy. We will merge directly into initial_dataset_path.
    print(f"Starting merge process. Target for merge operations: {initial_dataset_path}")
    print(f"Data from {extended_dataset_path} will be merged into {initial_dataset_path}.")
    print(f"Finally, {initial_dataset_path} will be renamed to {final_merged_dataset_target_path}.")

    for data_type in data_types:
        print(f"Processing {data_type}...")

        # Define paths for current data type
        # Data will be merged into the initial_dataset_path's subdirectories
        current_initial_data_dir = os.path.join(initial_dataset_path, data_type, "data")
        current_extended_data_dir = os.path.join(extended_dataset_path, data_type, "data")

        current_initial_timestamps_file = os.path.join(initial_dataset_path, data_type, "timestamps.txt")
        current_extended_timestamps_file = os.path.join(extended_dataset_path, data_type, "timestamps.txt")

        # Ensure the target data directory exists in the initial dataset
        os.makedirs(current_initial_data_dir, exist_ok=True)

        # Move data files from extended to initial
        if os.path.exists(current_extended_data_dir):
            print(f"Moving data files from {current_extended_data_dir} to {current_initial_data_dir}...")
            for filename in os.listdir(current_extended_data_dir):
                source_file_path = os.path.join(current_extended_data_dir, filename)
                destination_file_path = os.path.join(current_initial_data_dir, filename)
                shutil.move(source_file_path, destination_file_path)
            print(f"Data files for {data_type} moved.")
        else:
            print(f"Data directory {current_extended_data_dir} not found. Skipping file move for {data_type}.")


        # Merge timestamps.txt directly into the initial dataset's timestamps.txt
        print(f"Attempting to merge timestamps for {data_type} into {current_initial_timestamps_file}...")

        initial_exists = os.path.exists(current_initial_timestamps_file)
        extended_exists = os.path.exists(current_extended_timestamps_file)

        if initial_exists and extended_exists:
            # Both exist, append extended to initial
            print(f"Both timestamp files found. Merging {current_extended_timestamps_file} into {current_initial_timestamps_file}.")
            # Ensure a newline character if the initial file doesn't end with one and is not empty
            if os.path.getsize(current_initial_timestamps_file) > 0:
                needs_newline = False
                with open(current_initial_timestamps_file, 'rb') as f_read: # Use 'rb' to correctly handle tell() and seek() with last byte
                    f_read.seek(-1, os.SEEK_END) # Go to the last byte
                    if f_read.read(1) != b'\\n':
                        needs_newline = True
                if needs_newline:
                    with open(current_initial_timestamps_file, 'a') as outfile_newline: # Open in append mode to add newline
                        outfile_newline.write('\\n')

            with open(current_initial_timestamps_file, 'a') as outfile_append: # Open in append mode for main content
                with open(current_extended_timestamps_file, 'r') as infile:
                    outfile_append.write(infile.read())
            print(f"Timestamps from {current_extended_timestamps_file} successfully appended to {current_initial_timestamps_file}.")
        elif not initial_exists:
            # Initial timestamps file is missing. No merge possible.
            print(f"Warning: Initial timestamps file {current_initial_timestamps_file} not found. Timestamps for {data_type} cannot be created or merged.")
            if extended_exists:
                 print(f"   (Note: Extended file {current_extended_timestamps_file} was found but will not be used as the primary/initial file is missing).")
        elif not extended_exists: # This implies initial_exists is True
            # Initial exists, but extended is missing. No merge performed.
            print(f"Warning: Extended timestamps file {current_extended_timestamps_file} not found. The existing initial timestamps file {current_initial_timestamps_file} will be used as is for {data_type}. No merge performed.")
        # If both are missing, the 'not initial_exists' case handles the primary warning.

    print("Processing metadata...")
    # Define metadata paths
    current_initial_metadata_dir = os.path.join(initial_dataset_path, "metadata")
    current_extended_metadata_dir = os.path.join(extended_dataset_path, "metadata")

    # Ensure the target metadata directory exists
    os.makedirs(current_initial_metadata_dir, exist_ok=True)

    # Copy other metadata files (like os1.json) from extended if they don't exist in initial
    if os.path.exists(current_extended_metadata_dir):
        print(f"Checking for additional metadata files in {current_extended_metadata_dir}...")
        for filename in os.listdir(current_extended_metadata_dir):
            # Exclude the specific pkl file that will be merged
            if filename != "dataset_ext_indices.pkl":
                source_file = os.path.join(current_extended_metadata_dir, filename)
                target_file = os.path.join(current_initial_metadata_dir, filename)
                if not os.path.exists(target_file):
                    print(f"Copying {source_file} to {target_file} as it does not exist in target.")
                    shutil.copy2(source_file, target_file)
                else:
                    print(f"Skipping {filename} from extended, as it already exists in {current_initial_metadata_dir}.")

    # Merge dataset indices files
    # The target for the merged indices will be dataset_comp_indices.pkl in the initial dataset's metadata.
    target_indices_file = os.path.join(current_initial_metadata_dir, "dataset_comp_indices.pkl")
    source_initial_indices_file = os.path.join(current_initial_metadata_dir, "dataset_indices.pkl") # Path to initial indices
    source_extended_indices_file = os.path.join(current_extended_metadata_dir, "dataset_ext_indices.pkl")

    data1 = None
    data2 = None
    can_merge_indices = True # Flag to track if merging is possible

    print(f"Attempting to load initial dataset indices from {source_initial_indices_file}...")
    if not os.path.exists(source_initial_indices_file):
        print(f"Warning: Initial dataset indices file {source_initial_indices_file} not found. Cannot merge indices.")
        can_merge_indices = False
    else:
        try:
            with open(source_initial_indices_file, 'rb') as f:
                data1 = pickle.load(f)
            print("Successfully loaded initial dataset indices.")
        except Exception as e:
            print(f"Warning: Error loading initial dataset indices from {source_initial_indices_file}: {e}. Cannot merge indices.")
            can_merge_indices = False

    # Proceed to check the extended file, even if the first one failed, to provide complete warnings.
    # The can_merge_indices flag will ensure no merge happens if any step fails.
    print(f"Attempting to load extended dataset indices from {source_extended_indices_file}...")
    if not os.path.exists(source_extended_indices_file):
        print(f"Warning: Extended dataset indices file {source_extended_indices_file} not found. Cannot merge indices.")
        can_merge_indices = False # If already False, stays False.
    else:
        try:
            with open(source_extended_indices_file, 'rb') as f:
                data2 = pickle.load(f)
            print("Successfully loaded extended dataset indices.")
        except Exception as e:
            print(f"Warning: Error loading extended dataset indices from {source_extended_indices_file}: {e}. Cannot merge indices.")
            can_merge_indices = False # If already False, stays False.

    if data1 is not None and data2 is not None:
        # Both files were successfully loaded
        print("Both initial and extended dataset indices found and loaded. Merging...")
        merged_indices_data = data1.copy()

        # Calculate the offset needed for the extended dataset indices
        # Use the maximum index value from the original dataset plus 1
        initial_dataset_length = max(
            np.max(data1['train_indices']) if len(data1['train_indices']) > 0 else -1,
            np.max(data1['test_indices']) if len(data1['test_indices']) > 0 else -1
        ) + 1
        print(f"Initial dataset length: {initial_dataset_length} (will be used as offset for extended dataset indices)")

        for key, value_ext in data2.items(): # value_ext is from the extended dataset (data2)
            if key in merged_indices_data:
                value_init = merged_indices_data[key] # value_init is from the initial dataset (data1)
                if isinstance(value_init, list) and isinstance(value_ext, list):
                    print(f"  Merging list for key: '{key}'")
                    # If the list contains indices, offset them
                    if all(isinstance(x, (int, np.integer)) for x in value_ext):
                        value_ext = [x + initial_dataset_length for x in value_ext]
                    value_init.extend(value_ext)
                elif isinstance(value_init, np.ndarray) and isinstance(value_ext, np.ndarray):
                    print(f"  Concatenating numpy arrays for key: '{key}'")
                    # For numpy arrays containing indices, offset them
                    if np.issubdtype(value_ext.dtype, np.integer):
                        print(f"    Offsetting indices in extended dataset by {initial_dataset_length}")
                        value_ext = value_ext + initial_dataset_length
                    
                    # Ensure they are primarily 1D arrays or compatible for simple concatenation along axis 0
                    if value_init.ndim == value_ext.ndim:
                        try:
                            merged_indices_data[key] = np.concatenate((value_init, value_ext), axis=0)
                            print(f"    Successfully merged arrays for '{key}'. New shape: {merged_indices_data[key].shape}")
                        except ValueError as e:
                            print(f"    Warning: Could not concatenate numpy arrays for key '{key}': {e}. Overwriting with value from extended dataset.")
                            merged_indices_data[key] = value_ext
                    else:
                        print(f"    Warning: Numpy arrays for key '{key}' have different dimensions ({value_init.ndim} vs {value_ext.ndim}). Overwriting with value from extended dataset.")
                        merged_indices_data[key] = value_ext
                else:
                    print(f"  Warning: Key '{key}' exists in both datasets. Initial type: {type(value_init)}, Extended type: {type(value_ext)}. Overwriting with value from extended dataset for this key.")
                    merged_indices_data[key] = value_ext
            else:
                # For new keys from extended dataset, still need to offset if they contain indices
                if isinstance(value_ext, np.ndarray) and np.issubdtype(value_ext.dtype, np.integer):
                    value_ext = value_ext + initial_dataset_length
                elif isinstance(value_ext, list) and all(isinstance(x, (int, np.integer)) for x in value_ext):
                    value_ext = [x + initial_dataset_length for x in value_ext]
                merged_indices_data[key] = value_ext
        print("Dataset indices merged.")
    
        print(f"Saving merged dataset indices to {target_indices_file}...")
        try:
            with open(target_indices_file, 'wb') as f:
                pickle.dump(merged_indices_data, f)
            print("Merged dataset indices saved successfully.")
        except Exception as e:
            print(f"Error saving merged dataset indices to {target_indices_file}: {e}")
            print("Dataset indices merging failed at the saving stage. The target file may not have been updated.")

    else:
        # At least one of the pkl files was not found or could not be loaded.
        # Detailed reasons for data1/data2 being None would have been printed during their respective loading attempts.
        reason_summary = ""
        if data1 is None and data2 is None:
            reason_summary = f"neither the initial ({source_initial_indices_file}) nor the extended ({source_extended_indices_file}) dataset indices could be loaded"
        elif data1 is None:
            reason_summary = f"the initial dataset indices ({source_initial_indices_file}) could not be loaded"
        elif data2 is None: # This is the only remaining case
            reason_summary = f"the extended dataset indices ({source_extended_indices_file}) could not be loaded"
        else: # Should not be reached if logic is correct, but as a fallback
            reason_summary = "one or both dataset indices files could not be loaded (unexpected state)"

        print(f"Warning: Dataset indices merging skipped because {reason_summary}.")
        print(f"   The target file {target_indices_file} will not be created or modified, preserving its current state if it exists.")

    # Clean up empty "data" directories in the extended dataset path
    print(f"Cleaning up {extended_dataset_path}...")
    for data_type_to_clean in data_types:
        dir_to_check = os.path.join(extended_dataset_path, data_type_to_clean, "data")
        if os.path.exists(dir_to_check) and not os.listdir(dir_to_check):
            print(f"Removing empty data directory: {dir_to_check}")
            os.rmdir(dir_to_check)
        
        parent_dir_to_check = os.path.join(extended_dataset_path, data_type_to_clean)
        if os.path.exists(parent_dir_to_check) and not os.listdir(parent_dir_to_check):
             print(f"Removing empty type directory: {parent_dir_to_check} as it's now completely empty.")
             os.rmdir(parent_dir_to_check)
        # If parent_dir_to_check still contains timestamps.txt, it won't be removed by the above, which is intended.

    # Rename the initial dataset path to the final merged name
    print(f"Attempting to rename {initial_dataset_path} to {final_merged_dataset_target_path}...")
    if os.path.abspath(initial_dataset_path) == os.path.abspath(final_merged_dataset_target_path):
        print(f"Source and target for rename are the same ({initial_dataset_path}). No rename needed.")
        print("Dataset merging process complete.")
        print(f"Final dataset is located at: {initial_dataset_path}")
    elif os.path.exists(final_merged_dataset_target_path):
        print(f"Error: Target path {final_merged_dataset_target_path} already exists. Please remove or rename it manually before running the script again.")
        print(f"Merging process halted before final rename. Merged data is in: {initial_dataset_path}")
    else:
        try:
            os.rename(initial_dataset_path, final_merged_dataset_target_path)
            print(f"Successfully renamed {initial_dataset_path} to {final_merged_dataset_target_path}.")
            print("Dataset merging process complete.")
            print(f"Final dataset is located at: {final_merged_dataset_target_path}")
        except OSError as e:
            print(f"Error renaming directory: {e}")
            print(f"The merged data is currently in: {initial_dataset_path}")
