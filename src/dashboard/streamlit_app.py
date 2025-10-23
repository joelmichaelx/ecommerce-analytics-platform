"""
Streamlit App Entry Point
Main entry point for the E-commerce Analytics Dashboard
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from dashboard.main_dashboard import EcommerceDashboard

def main():
    """Main function to run the Streamlit dashboard"""
    try:
        # Initialize and run dashboard
        dashboard = EcommerceDashboard()
        dashboard.run_dashboard()
        
    except Exception as e:
        st.error(f"Failed to start dashboard: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
