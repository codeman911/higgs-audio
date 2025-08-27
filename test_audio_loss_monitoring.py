#!/usr/bin/env python3
"""
Test script for the Audio Loss Monitoring system.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trainer import AudioLossMonitor


def test_audio_loss_monitor():
    """Test the AudioLossMonitor class."""
    print("Testing AudioLossMonitor...")
    
    # Create monitor with default settings
    monitor = AudioLossMonitor(window_size=10, threshold=0.01, critical_threshold=0.005)
    
    # Test normal loss values
    print("\n1. Testing normal loss values:")
    for i in range(5):
        loss = 1.0 - (i * 0.1)  # Decreasing from 1.0 to 0.6
        alerts = monitor.update(loss, i)
        print(f"  Step {i}: Loss = {loss:.4f}, Alerts = {len(alerts)}")
    
    # Test low loss values that should trigger warnings
    print("\n2. Testing low loss values (should trigger warnings):")
    for i in range(5, 10):
        loss = 0.015 - ((i - 5) * 0.002)  # Decreasing from 0.015 to 0.007
        alerts = monitor.update(loss, i)
        print(f"  Step {i}: Loss = {loss:.4f}, Alerts = {len(alerts)}")
        for alert in alerts:
            print(f"    Alert: {alert['message']}")
    
    # Test critical low loss values that should trigger critical alerts
    print("\n3. Testing critical low loss values (should trigger critical alerts):")
    for i in range(10, 15):
        loss = 0.006 - ((i - 10) * 0.001)  # Decreasing from 0.006 to 0.001
        alerts = monitor.update(loss, i)
        print(f"  Step {i}: Loss = {loss:.4f}, Alerts = {len(alerts)}")
        for alert in alerts:
            print(f"    Alert: {alert['message']}")
    
    # Test recovery
    print("\n4. Testing recovery (should reset alerts):")
    for i in range(15, 20):
        loss = 0.02 + ((i - 15) * 0.01)  # Increasing from 0.02 to 0.06
        alerts = monitor.update(loss, i)
        print(f"  Step {i}: Loss = {loss:.4f}, Alerts = {len(alerts)}")
    
    # Test sustained low loss
    print("\n5. Testing sustained low loss (should trigger sustained warning):")
    for i in range(20, 35):
        loss = 0.008  # Constant low loss
        alerts = monitor.update(loss, i)
        print(f"  Step {i}: Loss = {loss:.4f}, Alerts = {len(alerts)}")
        for alert in alerts:
            print(f"    Alert: {alert['message']}")
    
    # Test diagnostics
    print("\n6. Testing diagnostics:")
    diagnostics = monitor.get_detailed_diagnostics()
    print(f"  Diagnostics: {diagnostics}")
    
    # Test recovery attempts
    print("\n7. Testing recovery attempts:")
    monitor.record_recovery_attempt(30)
    diagnostics = monitor.get_detailed_diagnostics()
    print(f"  Diagnostics after recovery attempt: {diagnostics}")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_audio_loss_monitor()