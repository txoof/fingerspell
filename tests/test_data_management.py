"""
Unit tests for data_management module.

Tests save and discard functionality.
"""

import pytest
import csv
import tempfile
from pathlib import Path
from src.fingerspell.collection.data_management import save_final_data


class TestSaveFinalData:
    """Tests for save_final_data function."""
    
    def test_strips_sample_id_column(self):
        """Should remove sample_id (first column) from output."""
        # Create temp file with sample_id
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        csv_writer = csv.writer(temp_file)
        
        # Write test data: sample_id, label, features...
        test_data = [
            [0, 0, 0.1, 0.2, 0.3],
            [1, 1, 0.4, 0.5, 0.6],
            [2, 0, 0.7, 0.8, 0.9],
        ]
        csv_writer.writerows(test_data)
        temp_file.close()
        
        # Test alphabet and mapping
        alphabet = ['A', 'B']
        label_map = {'A': 0, 'B': 1}
        
        # Save (this will create files on Desktop, we'll clean up)
        result = save_final_data(temp_file.name, alphabet, label_map)
        
        assert result == True
        
        # Find the created directory
        desktop = Path.home() / "Desktop"
        created_dirs = sorted(desktop.glob("fingerspell_data_*"))
        assert len(created_dirs) > 0
        
        latest_dir = created_dirs[-1]
        output_file = latest_dir / "keypoint_data.csv"
        
        # Verify sample_id was stripped
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Should have 4 columns (label + 3 features), not 5
        assert len(rows[0]) == 4
        assert rows[0] == ['0', '0.1', '0.2', '0.3']
        
        # Cleanup
        import shutil
        shutil.rmtree(latest_dir)
        Path(temp_file.name).unlink()
    
    def test_handles_mixed_row_lengths(self):
        """Should handle static (44 cols) and dynamic (86 cols) rows."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        csv_writer = csv.writer(temp_file)
        
        # Static row: sample_id, label, 42 features
        static_row = [0, 0] + [0.1] * 42
        
        # Dynamic row: sample_id, label, 84 features
        dynamic_row = [1, 3] + [0.2] * 84
        
        csv_writer.writerow(static_row)
        csv_writer.writerow(dynamic_row)
        temp_file.close()
        
        alphabet = ['A', 'H']
        label_map = {'A': 0, 'H': 3}
        
        result = save_final_data(temp_file.name, alphabet, label_map)
        
        assert result == True
        
        # Find and check output
        desktop = Path.home() / "Desktop"
        created_dirs = sorted(desktop.glob("fingerspell_data_*"))
        latest_dir = created_dirs[-1]
        output_file = latest_dir / "keypoint_data.csv"
        
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # First row should have 43 columns (label + 42)
        assert len(rows[0]) == 43
        
        # Second row should have 85 columns (label + 84)
        assert len(rows[1]) == 85
        
        # Cleanup
        import shutil
        shutil.rmtree(latest_dir)
        Path(temp_file.name).unlink()
    
    def test_creates_label_file(self):
        """Should create keypoint_classifier_label.csv file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        csv_writer = csv.writer(temp_file)
        csv_writer.writerow([0, 0, 0.1, 0.2])
        temp_file.close()
        
        alphabet = ['C', 'A', 'B']  # Unsorted
        label_map = {'A': 0, 'B': 1, 'C': 2}  # Sorted by unicode
        
        result = save_final_data(temp_file.name, alphabet, label_map)
        
        assert result == True
        
        # Find output directory
        desktop = Path.home() / "Desktop"
        created_dirs = sorted(desktop.glob("fingerspell_data_*"))
        latest_dir = created_dirs[-1]
        label_file = latest_dir / "keypoint_classifier_label.csv"
        
        # Verify label file exists and is sorted
        assert label_file.exists()
        
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            labels = [row[0] for row in reader]
        
        # Should be sorted by label index (A=0, B=1, C=2)
        assert labels == ['A', 'B', 'C']
        
        # Cleanup
        import shutil
        shutil.rmtree(latest_dir)
        Path(temp_file.name).unlink()
    
    def test_creates_timestamped_directory(self):
        """Should create directory with timestamp on Desktop."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        csv_writer = csv.writer(temp_file)
        csv_writer.writerow([0, 0, 0.1])
        temp_file.close()
        
        alphabet = ['A']
        label_map = {'A': 0}
        
        # Count existing directories
        desktop = Path.home() / "Desktop"
        before_count = len(list(desktop.glob("fingerspell_data_*")))
        
        result = save_final_data(temp_file.name, alphabet, label_map)
        
        assert result == True
        
        # Should have one more directory
        after_count = len(list(desktop.glob("fingerspell_data_*")))
        assert after_count == before_count + 1
        
        # Verify directory name format
        created_dirs = sorted(desktop.glob("fingerspell_data_*"))
        latest_dir = created_dirs[-1]
        assert latest_dir.name.startswith("fingerspell_data_")
        assert len(latest_dir.name) == len("fingerspell_data_YYYYMMDD_HHMM")
        
        # Cleanup
        import shutil
        shutil.rmtree(latest_dir)
        Path(temp_file.name).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
