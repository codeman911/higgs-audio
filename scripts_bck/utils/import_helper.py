#!/usr/bin/env python3
"""
Universal Import Helper for Higgs-Audio Scripts
Provides robust import handling for both CLI and module usage
"""

import os
import sys
from pathlib import Path


def setup_project_imports():
    """
    Setup project imports by adding the project root to Python path.
    This function handles various project structures and execution contexts.
    """
    # Get the directory containing this file
    current_file = Path(__file__).resolve()
    
    # Navigate up to find project root (contains boson_multimodal)
    project_root = None
    search_path = current_file.parent
    
    # Search up the directory tree for project root
    for _ in range(5):  # Limit search depth
        if (search_path / "boson_multimodal").exists():
            project_root = search_path
            break
        search_path = search_path.parent
    
    # Fallback: assume project root is 2 levels up from scripts/utils/
    if project_root is None:
        project_root = current_file.parent.parent.parent
    
    # Add project root to Python path if not already present
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def safe_import(module_name, fallback_paths=None):
    """
    Safely import a module with fallback paths.
    
    Args:
        module_name (str): Name of the module to import
        fallback_paths (list): Additional paths to try if import fails
    
    Returns:
        module: The imported module
    
    Raises:
        ImportError: If module cannot be imported from any path
    """
    # Try direct import first
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError:
        pass
    
    # Setup project imports
    setup_project_imports()
    
    # Try import again after setting up paths
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError:
        pass
    
    # Try fallback paths if provided
    if fallback_paths:
        original_path = sys.path.copy()
        for path in fallback_paths:
            try:
                if path not in sys.path:
                    sys.path.insert(0, str(path))
                return __import__(module_name, fromlist=[''])
            except ImportError:
                continue
            finally:
                sys.path = original_path
    
    # If all attempts fail, raise ImportError
    raise ImportError(f"Could not import {module_name} from any available path")


# Convenience function for common Higgs-Audio imports
def import_higgs_audio_modules():
    """Import common Higgs-Audio modules with error handling"""
    setup_project_imports()
    
    modules = {}
    
    try:
        from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
        modules['data_types'] = {
            'ChatMLSample': ChatMLSample,
            'Message': Message,
            'AudioContent': AudioContent,
            'TextContent': TextContent
        }
    except ImportError as e:
        print(f"Warning: Could not import data_types: {e}")
    
    try:
        from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioForCausalLM
        from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
        modules['model'] = {
            'HiggsAudioForCausalLM': HiggsAudioForCausalLM,
            'HiggsAudioConfig': HiggsAudioConfig
        }
    except ImportError as e:
        print(f"Warning: Could not import model modules: {e}")
    
    try:
        from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
        modules['dataset'] = {
            'ChatMLDatasetSample': ChatMLDatasetSample,
            'prepare_chatml_sample': prepare_chatml_sample
        }
    except ImportError as e:
        print(f"Warning: Could not import dataset modules: {e}")
    
    return modules


if __name__ == "__main__":
    # Test the import helper
    print("Testing Higgs-Audio Import Helper")
    print("=" * 40)
    
    project_root = setup_project_imports()
    print(f"Project root: {project_root}")
    print(f"Python path entries: {len(sys.path)}")
    
    modules = import_higgs_audio_modules()
    print(f"Successfully imported {len(modules)} module groups:")
    for name, items in modules.items():
        print(f"  - {name}: {list(items.keys())}")
