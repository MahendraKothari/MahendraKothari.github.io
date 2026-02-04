"""
TACTIC Pointing Correction - Streamlit Web Version
This allows users to run the software from any browser without installation
"""

import streamlit as st
import os
import numpy as np
from io import BytesIO
import tempfile
from TACTIC_Pointing_Correction import TACTICPointingCorrection

st.set_page_config(
    page_title="TACTIC Pointing Correction",
    page_icon="🔭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        color: #4A90E2;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔭 TACTIC Pointing Correction</div>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Source Configuration
st.sidebar.subheader("🎯 Source Details")
source_name = st.sidebar.text_input("Source Name", "MU_Geminorum")
source_ra = st.sidebar.number_input("RA (hours)", value=5.627411, format="%.6f")
source_dec = st.sidebar.number_input("DEC (degrees)", value=21.142584, format="%.6f")

# Observatory Configuration
st.sidebar.subheader("🌍 Observatory")
obs_lat = st.sidebar.number_input("Latitude", value=24.6548, format="%.4f")
obs_lon = st.sidebar.number_input("Longitude", value=72.7792, format="%.4f")
obs_height = st.sidebar.number_input("Height (m)", value=1360)

# File Upload Section
st.header("📁 Upload Files")

col1, col2 = st.columns(2)

with col1:
    st.subheader("FITS Files")
    fits_files = st.file_uploader(
        "Upload FITS files (.fit)", 
        accept_multiple_files=True,
        type=['fit', 'fits']
    )
    
with col2:
    st.subheader("Timestamp File")
    timestamp_file = st.file_uploader(
        "Upload timestamp file (.txt)",
        type=['txt']
    )

# Reference Points
st.header("🎯 Reference Points")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("LED Coordinates")
    led_a_x = st.number_input("LED A - X", value=546.0919, format="%.4f")
    led_a_y = st.number_input("LED A - Y", value=546.2835, format="%.4f")
    led_b_x = st.number_input("LED B - X", value=543.5715, format="%.4f")
    led_b_y = st.number_input("LED B - Y", value=119.8698, format="%.4f")

with col2:
    led_c_x = st.number_input("LED C - X", value=332.5692, format="%.4f")
    led_c_y = st.number_input("LED C - Y", value=336.5287, format="%.4f")
    led_d_x = st.number_input("LED D - X", value=755.4114, format="%.4f")
    led_d_y = st.number_input("LED D - Y", value=335.6598, format="%.4f")

with col3:
    st.subheader("Star Coordinates")
    star_pre_x = st.number_input("Star Pre - X", value=554.2152, format="%.4f")
    star_pre_y = st.number_input("Star Pre - Y", value=301.6761, format="%.4f")
    star_post_x = st.number_input("Star Post - X", value=0.0, format="%.4f")
    star_post_y = st.number_input("Star Post - Y", value=0.0, format="%.4f")

# Polynomial Fitting Options
st.header("📊 Polynomial Fitting Options")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pre-Transit Orders")
    pre_orders = st.multiselect(
        "Select orders",
        [1, 2, 3, 4, 5],
        default=[1, 2, 3]
    )

with col2:
    st.subheader("Post-Transit Orders")
    post_orders = st.multiselect(
        "Select orders",
        [1, 2, 3, 4, 5],
        default=[1, 2, 3]
    )

# Analysis Button
if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
    if not fits_files or not timestamp_file:
        st.error("❌ Please upload FITS files and timestamp file!")
    else:
        with st.spinner("🔄 Processing..."):
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save uploaded files
                fits_dir = os.path.join(tmpdir, "fits")
                os.makedirs(fits_dir, exist_ok=True)
                
                for fits_file in fits_files:
                    with open(os.path.join(fits_dir, fits_file.name), 'wb') as f:
                        f.write(fits_file.read())
                
                timestamp_path = os.path.join(tmpdir, "timestamps.txt")
                with open(timestamp_path, 'wb') as f:
                    f.write(timestamp_file.read())
                
                # Initialize analyzer
                analyzer = TACTICPointingCorrection()
                analyzer.SOURCE_NAME = source_name
                analyzer.SOURCE_RA = source_ra * 15
                analyzer.SOURCE_DEC = source_dec
                analyzer.OBSERVATORY_LAT = obs_lat
                analyzer.OBSERVATORY_LON = obs_lon
                analyzer.OBSERVATORY_HEIGHT = obs_height
                analyzer.fits_dir = fits_dir
                analyzer.timestamp_file = timestamp_path
                analyzer.output_dir = tmpdir
                
                # Set reference points
                analyzer.reference_points = {
                    'A_up': (led_a_x, led_a_y),
                    'B_down': (led_b_x, led_b_y),
                    'C_left': (led_c_x, led_c_y),
                    'D_right': (led_d_x, led_d_y),
                    'Star_Pre': (star_pre_x, star_pre_y),
                    'Star_Post': (star_post_x, star_post_y),
                    'Star': (star_pre_x, star_pre_y)
                }
                
                try:
                    # Run analysis
                    st.info("📊 Parsing timestamps...")
                    timestamps = analyzer.parse_timestamps()
                    
                    st.info("🔍 Processing FITS files...")
                    df = analyzer.process_fits_files(timestamps)
                    
                    st.info("📈 Creating plots...")
                    analyzer.plot_fits(df, pre_orders=pre_orders, post_orders=post_orders)
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display results
                    st.header("📊 Results")
                    
                    # Show dataframe
                    st.subheader("Data Table")
                    st.dataframe(df)
                    
                    # Download Excel
                    excel_path = os.path.join(tmpdir, "pointing_correction_results.xlsx")
                    df.to_excel(excel_path, index=False)
                    
                    with open(excel_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Excel Results",
                            f.read(),
                            "pointing_correction_results.xlsx",
                            "application/vnd.ms-excel"
                        )
                    
                    # Show plots
                    plot_path = os.path.join(tmpdir, "pointing_fits_pre_post.png")
                    if os.path.exists(plot_path):
                        st.subheader("Pre/Post Transit Plots")
                        st.image(plot_path)
                        
                        with open(plot_path, 'rb') as f:
                            st.download_button(
                                "📥 Download Plot",
                                f.read(),
                                "pointing_fits_pre_post.png",
                                "image/png"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "TACTIC Pointing Correction Tool | Developed for Astronomical Data Analysis"
    "</div>",
    unsafe_allow_html=True
)
