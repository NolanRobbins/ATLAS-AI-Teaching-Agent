
import unittest
from unittest.mock import patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestStreamlitApp(unittest.TestCase):
    @patch('streamlit.write')
    def test_main_app_runs_without_error(self, mock_write):
        # This is a basic "smoke test" to ensure the app runs without crashing.
        # It doesn't test the full functionality, but it's a good start.
        from app import main_app
        
        # Mocking Streamlit's behavior
        with patch('streamlit.text_input', return_value='test'), \
             patch('streamlit.selectbox', return_value='Freshman'), \
             patch('streamlit.checkbox', return_value=True), \
             patch('streamlit.number_input', return_value=1), \
             patch('streamlit.date_input'), \
             patch('streamlit.time_input'), \
             patch('streamlit.text_area', return_value='test request'), \
             patch('streamlit.button', return_value=True), \
             patch('streamlit.spinner'), \
             patch('app.get_openai_key', return_value='fake_key'), \
             patch('app.run_all_system_streamlit', return_value=(None, None)):            
            main_app()
        
        # Verify that some Streamlit functions were called
        self.assertTrue(mock_write.called)

if __name__ == '__main__':
    unittest.main()
