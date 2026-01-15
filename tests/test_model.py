from unittest.mock import Mock


def test_example_with_mock():
    """Test example using a mock object."""
    mock_function = Mock(return_value=42)
    result = mock_function()
    assert result == 42
    mock_function.assert_called_once()
