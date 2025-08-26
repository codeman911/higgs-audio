#!/usr/bin/env python3
"""
Test Script for Fixed Validation Logic

Run this on your remote cluster to verify the validation fix works correctly.

Usage:
    python3 test_validation_fix.py --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any


def test_validation_logic(data_path: str, max_samples: int = None) -> Dict[str, Any]:
    """
    Test the fixed validation logic on actual training data.
    
    Args:
        data_path: Path to the training data file
        max_samples: Maximum number of samples to test (None = all)
        
    Returns:
        Dictionary with test results
    """
    results = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "error_samples": 0,
        "validation_errors": [],
        "sample_structures": [],
        "processing_time": 0
    }
    
    start_time = time.time()
    
    try:
        print(f"🔍 Testing validation on: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"❌ Training data file not found: {data_path}")
            return results
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        results["total_samples"] = len(data)
        
        # Limit samples for testing if specified
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
            print(f"📊 Testing on first {max_samples} of {results['total_samples']} samples")
        else:
            print(f"📊 Testing on all {len(data)} samples")
        
        # Test validation on each sample
        for idx, sample in enumerate(data):
            try:
                is_valid = _validate_sample_structure_fixed(sample, idx, verbose=(idx < 5))
                
                if is_valid:
                    results["valid_samples"] += 1
                    
                    # Collect sample structure info for first few samples
                    if idx < 3:
                        structure_info = _analyze_sample_structure(sample, idx)
                        results["sample_structures"].append(structure_info)
                else:
                    results["invalid_samples"] += 1
                    
                # Progress indicator for large datasets
                if (idx + 1) % 10000 == 0:
                    print(f"   Progress: {idx + 1}/{len(data)} samples processed...")
                    
            except Exception as e:
                results["error_samples"] += 1
                error_info = {
                    "sample_idx": idx,
                    "error": str(e),
                    "sample_preview": str(sample)[:200] + "..." if len(str(sample)) > 200 else str(sample)
                }
                results["validation_errors"].append(error_info)
                
                if len(results["validation_errors"]) <= 5:  # Show first 5 errors
                    print(f"   ❌ Sample {idx}: Validation error: {e}")
        
        results["processing_time"] = time.time() - start_time
        
        # Print summary
        print(f"\n✅ Validation Test Results:")
        print(f"   📊 Total samples tested: {len(data)}")
        print(f"   ✅ Valid samples: {results['valid_samples']}")
        print(f"   ❌ Invalid samples: {results['invalid_samples']}")
        print(f"   💥 Error samples: {results['error_samples']}")
        print(f"   ⏱️ Processing time: {results['processing_time']:.2f} seconds")
        print(f"   📈 Success rate: {(results['valid_samples'] / len(data)) * 100:.1f}%")
        
        if results["validation_errors"]:
            print(f"\n❌ Sample validation errors ({len(results['validation_errors'])} total):")
            for error in results["validation_errors"][:3]:  # Show first 3 errors
                print(f"   Sample {error['sample_idx']}: {error['error']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to test validation: {e}")
        results["processing_time"] = time.time() - start_time
        return results


def _validate_sample_structure_fixed(sample: Dict[str, Any], idx: int, verbose: bool = False) -> bool:
    """FIXED validation using exact arb_inference.py process_chatml_sample logic."""
    try:
        # Check required fields
        if "messages" not in sample:
            if verbose:
                print(f"  Sample {idx}: Missing 'messages' field")
            return False
        
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            if verbose:
                print(f"  Sample {idx}: Invalid 'messages' field")
            return False
        
        # Use EXACT same logic as arb_inference.py process_chatml_sample
        ref_audio_path = None
        ref_text = None
        target_text = None
        
        for message in messages:
            if not isinstance(message, dict):
                if verbose:
                    print(f"  Sample {idx}: Invalid message format")
                return False
            
            if "role" not in message or "content" not in message:
                if verbose:
                    print(f"  Sample {idx}: Missing 'role' or 'content'")
                return False
            
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    # Look for text and audio content (EXACT arb_inference.py logic)
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item["text"])
                            elif item.get("type") == "audio":
                                if ref_audio_path is None:  # First audio is reference
                                    ref_audio_path = item.get("audio_url")
                    
                    if len(text_parts) >= 2:
                        ref_text = text_parts[0]  # First text is reference
                        # Look for target text
                        for text_part in text_parts[1:]:
                            if "Please generate speech" in text_part:
                                # Extract target text after the instruction
                                target_text = text_part.split(":")[-1].strip()
                                break
                        if target_text is None and len(text_parts) > 1:
                            target_text = text_parts[-1]  # Last text as fallback
                elif isinstance(content, str):
                    # Simple string content
                    if ref_text is None:
                        ref_text = content
                    else:
                        target_text = content
                        
            elif message["role"] == "assistant":
                content = message["content"]
                if isinstance(content, dict) and content.get("type") == "audio":
                    if ref_audio_path is None:
                        ref_audio_path = content.get("audio_url")
        
        # Validate that we found all required components (EXACT arb_inference.py logic)
        if not all([ref_audio_path, ref_text, target_text]):
            if verbose:
                print(f"  Sample {idx}: Missing required components:")
                print(f"    - ref_audio: {'✅' if ref_audio_path else '❌'} {ref_audio_path}")
                print(f"    - ref_text: {'✅' if ref_text else '❌'} {ref_text[:50] if ref_text else 'None'}...")
                print(f"    - target_text: {'✅' if target_text else '❌'} {target_text[:50] if target_text else 'None'}...")
            return False
        
        if verbose:
            print(f"  Sample {idx}: ✅ Valid structure")
            print(f"    - ref_audio: {ref_audio_path}")
            print(f"    - ref_text: '{ref_text[:50]}...'")
            print(f"    - target_text: '{target_text[:50]}...'")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"  Sample {idx}: Validation error: {e}")
        return False


def _analyze_sample_structure(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Analyze the structure of a sample for debugging."""
    structure = {
        "sample_idx": idx,
        "has_messages": "messages" in sample,
        "num_messages": len(sample.get("messages", [])),
        "message_roles": [],
        "content_types": [],
        "has_speaker": "speaker" in sample,
        "has_start_index": "start_index" in sample
    }
    
    for msg in sample.get("messages", []):
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content")
            
            structure["message_roles"].append(role)
            
            if isinstance(content, str):
                structure["content_types"].append("string")
            elif isinstance(content, list):
                structure["content_types"].append("list")
            elif isinstance(content, dict):
                if content.get("type") == "audio":
                    structure["content_types"].append("audio_dict")
                else:
                    structure["content_types"].append("other_dict")
            else:
                structure["content_types"].append("unknown")
    
    return structure


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Fixed Validation Logic")
    
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data file")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to test (default: all)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output for first few samples")
    
    args = parser.parse_args()
    
    print("🧪 Testing Fixed Validation Logic")
    print("=" * 50)
    print("🔧 This script tests the corrected validation logic")
    print("   that matches arb_inference.py patterns exactly.")
    print("=" * 50)
    
    # Test validation
    results = test_validation_logic(args.train_data, args.max_samples)
    
    # Determine success
    if results["total_samples"] == 0:
        print("❌ No samples found to test")
        sys.exit(1)
    elif results["valid_samples"] == 0:
        print("❌ No valid samples found - validation logic may still have issues")
        sys.exit(1)
    elif results["valid_samples"] > results["total_samples"] * 0.9:  # 90% success rate
        print("🎉 Validation test PASSED! Most samples are valid.")
        print("✅ Ready to proceed with training")
    else:
        print(f"⚠️ Validation test partially successful ({results['valid_samples']}/{results['total_samples']} valid)")
        print("💡 Some samples may need format conversion")
    
    # Save results
    results_file = "validation_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📋 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()