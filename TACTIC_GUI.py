import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import threading
import numpy as np
from datetime import datetime
from functools import partial
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Import your original class
from TACTIC_Pointing_Correction import TACTICPointingCorrection

class WorkerThread(QThread):
    """Background thread for analysis"""
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str, str)  # message, type
    finished = pyqtSignal()
    image_ready = pyqtSignal(object)  # For sending detection images
    indicator_update = pyqtSignal(str, bool)  # indicator_name, active
    
    def __init__(self, analyzer, pre_orders=None, post_orders=None):
        super().__init__()
        self.analyzer = analyzer
        self.pre_orders = pre_orders if pre_orders else [1, 2, 3]
        self.post_orders = post_orders if post_orders else [1, 2, 3]
        self._is_paused = False  
        self._mutex = QMutex()
        
    def pause(self):
        """Pause the analysis"""
        self._mutex.lock()
        self._is_paused = True
        self._mutex.unlock()
        self.log_message.emit("⏸️ Analysis paused", "info")
    
    def resume(self):
        """Resume the analysis"""
        self._mutex.lock()
        self._is_paused = False
        self._mutex.unlock()
        self.log_message.emit("▶️ Analysis resumed", "success")
    
    def is_paused(self):
        """Check if paused"""
        self._mutex.lock()
        paused = self._is_paused
        self._mutex.unlock()
        return paused
        
    def run(self):
        try:
            # Activate LED Detection indicator
            self.indicator_update.emit("led", True)
            self.log_message.emit("Starting analysis...", "info")
            self.log_message.emit("=" * 50, "info")
            
            self.log_message.emit("Parsing timestamps (IST → UTC)...", "info")
            self.progress.emit(10)
            timestamps = self.analyzer.parse_timestamps()
            self.log_message.emit(f"✓ Found {len(timestamps)} timestamps", "success")
            self.progress.emit(20)
            
            # Get FITS files
            fits_files = sorted([f for f in os.listdir(self.analyzer.fits_dir) if f.endswith('.fit')])
            self.log_message.emit(f"✓ Found {len(fits_files)} FITS files", "success")
            self.log_message.emit("=" * 50, "info")
            
            # Process FITS files with live image updates
            self.log_message.emit("Processing FITS files...", "info")
            self.progress.emit(30)
            
            # Activate Star Tracking indicator
            self.indicator_update.emit("star", True)
            
            # Custom process with image updates
            df = self.process_fits_with_updates(timestamps, fits_files)
            
            self.log_message.emit("=" * 50, "info")
            self.log_message.emit(f"✓ Processed {len(df)} images", "success")
            
            # TRANSIT REPORT
            transit_idx = df['zenith'].idxmin()
            transit_zenith = df.loc[transit_idx, 'zenith']
            transit_time = df.loc[transit_idx, 'Time']
            
            self.log_message.emit("=" * 50, "info")
            self.log_message.emit("🎯 UPPER TRANSIT DETECTED:", "success")
            self.log_message.emit(f"   Image Number: {transit_idx + 1}", "success")
            self.log_message.emit(f"   Zenith Angle: {transit_zenith:.2f}°", "success")
            self.log_message.emit(f"   Time (UTC): {transit_time}", "success")

            # Print Pre/Post Transit info
            print("\n" + "="*60)
            print("TRANSIT ANALYSIS SUMMARY")
            print("="*60)
            print(f"Transit at Image #{transit_idx + 1}")

            # Check Pre-Transit data
            if transit_idx > 0:
                pre_start = 1
                pre_end = transit_idx
                print(f"PRE-TRANSIT:  Images #{pre_start} to #{pre_end} ({pre_end} images)")
            else:
                print("PRE-TRANSIT:  No data available")

            # Check Post-Transit data
            if transit_idx < len(df) - 1:
                post_start = transit_idx + 2
                post_end = len(df)
                print(f"POST-TRANSIT: Images #{post_start} to #{post_end} ({post_end - post_start + 1} images)")
            else:
                print("POST-TRANSIT: No data available")

            print("="*60 + "\n")
            # Check if Star_Post was auto-detected
            star_post = self.analyzer.reference_points.get('Star_Post', (0, 0))
            if star_post != (0, 0):
                self.log_message.emit(f"   Star (Post): ({star_post[0]:.2f}, {star_post[1]:.2f}) ✓", "success")

            self.log_message.emit("=" * 50, "info")
            
            # Count successful star detections
            star_detected = df['Img_X'].notna().sum()
            star_failed = len(df) - star_detected
            self.log_message.emit(f"Star Detection: {star_detected} success, {star_failed} failed", "info")
            
            self.progress.emit(70)
            
            # Deactivate LED and Star indicators
            self.indicator_update.emit("led", False)
            self.indicator_update.emit("star", False)
            
            # Activate Zenith indicator
            self.indicator_update.emit("zenith", True)
            
            # Round all numeric columns
            self.log_message.emit("Formatting data (4 decimal places)...", "info")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Time':
                    df[col] = df[col].round(4)
            
            self.progress.emit(75)
            
            # Save Excel
            self.log_message.emit("Saving Excel results...", "info")
            excel_path = os.path.join(self.analyzer.output_dir, "pointing_correction_results.xlsx")
            df.to_excel(excel_path, index=False)
            self.log_message.emit(f"✓ Excel saved: {excel_path}", "success")
            self.progress.emit(80)
            
            # Create plots
            self.log_message.emit("Creating plots...", "info")
            self.log_message.emit("  - Analyzing pre-transit data...", "info")
            self.log_message.emit("  - Analyzing post-transit data...", "info")
            self.progress.emit(90)
            
            self.analyzer.plot_fits(df, pre_orders=self.pre_orders, post_orders=self.post_orders)
            
            plot_path = os.path.join(self.analyzer.output_dir, 'pointing_fits_pre_post.png')
            self.log_message.emit(f"✓ Plots saved: {plot_path}", "success")
            
            # Deactivate Zenith indicator
            self.indicator_update.emit("zenith", False)
            
            self.progress.emit(100)
            self.log_message.emit("=" * 50, "success")
            self.log_message.emit("✅ ANALYSIS COMPLETE!", "success")
            self.log_message.emit("=" * 50, "success")
            self.log_message.emit(f"All results saved in: {self.analyzer.output_dir}", "info")
            
        except Exception as e:
            self.log_message.emit("=" * 50, "error")
            self.log_message.emit(f"❌ Error: {str(e)}", "error")
            self.log_message.emit("=" * 50, "error")
            import traceback
            self.log_message.emit(traceback.format_exc(), "error")
        finally:
            # Reset all indicators
            self.indicator_update.emit("led", False)
            self.indicator_update.emit("star", False)
            self.indicator_update.emit("zenith", False)
            self.finished.emit()  
    
    def process_fits_with_updates(self, timestamps, fits_files):
        """Process FITS with simple, reliable star detection"""
        results = []
    
        # DETECT TRANSIT POINT
        self.log_message.emit("🔍 Detecting upper transit point...", "info")
    
        zenith_angles = []
        for i in range(min(len(timestamps), len(fits_files))):
            try:
                from astropy.coordinates import AltAz
                altaz = self.analyzer.target_coord.transform_to(
                    AltAz(obstime=timestamps[i], location=self.analyzer.location))
                zenith = 90.0 - altaz.alt.deg
                zenith_angles.append(zenith)
            except:
                zenith_angles.append(999)
    
        # Find transit (minimum zenith)
        transit_idx = np.argmin(zenith_angles)
        min_zenith = zenith_angles[transit_idx]
    
        # CLASSIFY PRE/POST
        has_pre = transit_idx > 2
        has_post = (len(fits_files) - transit_idx - 1) > 2
    
        self.log_message.emit(
            f"✓ Transit at Image #{transit_idx+1}, Zenith={min_zenith:.2f}°",
            "success"
        )
        
        # GET STAR REFERENCES
        star_pre_ref = self.analyzer.reference_points.get('Star_Pre', 
                       self.analyzer.reference_points.get('Star', (555.5624, 334.4543)))
        star_post_ref = self.analyzer.reference_points.get('Star_Post', (0, 0))
    
        # Check if Star_Post is provided
        has_star_post = (star_post_ref != (0, 0) and star_post_ref[0] != 0)
    
        if has_pre and has_post:
            if has_star_post:
                self.log_message.emit(f"  📊 Dataset: PRE + POST Transit", "info")
                self.log_message.emit(f"     Pre: Images 1-{transit_idx+1} using Star_Pre", "info")
                self.log_message.emit(f"     Post: Images {transit_idx+2}-{len(fits_files)} using Star_Post", "info")
            else:
                self.log_message.emit(f"  ⚠ Star_Post coordinates NOT provided!", "info")
                self.log_message.emit(f"     Will use Star_Pre for all images (may fail post-transit)", "info")
        elif has_pre:
            self.log_message.emit(f"  📊 Dataset: PRE-Transit ONLY", "info")
        elif has_post:
            self.log_message.emit(f"  📊 Dataset: POST-Transit ONLY", "info")
    
        # PROCESS FILES
        detection_failures = 0
    
        for i, fits_file in enumerate(fits_files):
            # Check if paused
            while self.is_paused():
                self.msleep(100)
        
            if i % 5 == 0:
                progress = 30 + int((i / len(fits_files)) * 35)
                self.progress.emit(progress)
            
                phase = "Pre-transit" if i <= transit_idx else "Post-transit"
                self.log_message.emit(f"Processing: {i+1}/{len(fits_files)} ({phase})", "info")
            
            fits_path = os.path.join(self.analyzer.fits_dir, fits_file)
        
            try:
                from astropy.io import fits
                with fits.open(fits_path) as hdul:
                    data = hdul[0].data.astype(float)
            except Exception as e:
                self.log_message.emit(f"⚠ Skip {fits_file}: {str(e)}", "error")
                continue
        
            # DETECT LEDs (always same coordinates)
            coords = {}
            # Track last good LED positions
            if not hasattr(self, 'last_led_positions'):
                self.last_led_positions = {}

            for name in ['A_up', 'B_down', 'C_left', 'D_right']:
                ref_x, ref_y = self.analyzer.reference_points[name]
                x_found, y_found = self.analyzer.find_led_star_advanced(data, ref_x, ref_y)
    
                # Use previous position if detection fails
                if x_found is None or np.isnan(x_found):
                    if name in self.last_led_positions:
                        coords[name] = self.last_led_positions[name]
                        if i % 20 == 0:  # Log occasionally
                            self.log_message.emit(f"  ⚠ LED {name}: Using previous position", "info")
                    else:
                        coords[name] = (np.nan, np.nan)
                else:
                    coords[name] = (x_found, y_found)
                    self.last_led_positions[name] = (x_found, y_found)  # Store good position
                ref_x, ref_y = self.analyzer.reference_points[name]
                x_found, y_found = self.analyzer.find_led_star_advanced(data, ref_x, ref_y)
                coords[name] = (x_found if x_found is not None else np.nan, 
                              y_found if y_found is not None else np.nan)
        
            # DETECT STAR - SIMPLE LOGIC
            if i <= transit_idx:
                # PRE-TRANSIT: Use Star_Pre
                star_ref = star_pre_ref
            else:
                # POST-TRANSIT: Use Star_Post if provided, else Star_Pre
                if has_star_post:
                    star_ref = star_post_ref
                else:
                    star_ref = star_pre_ref
        
            # Normal detection with standard search radius (50 pixels)
            star_x, star_y = self.analyzer.find_led_star_advanced(data, star_ref[0], star_ref[1])
        
            coords['Star'] = (star_x if star_x is not None else np.nan,
                             star_y if star_y is not None else np.nan)
        
            # Track failures
            if np.isnan(coords['Star'][0]):
                detection_failures += 1
                if detection_failures <= 5 or detection_failures % 10 == 0:
                    self.log_message.emit(f"  ⚠ Image {i+1}: Star not detected", "info")
        
            # Create detection image
            if i < self.analyzer.save_count:
                pixmap = self.create_detection_pixmap(data, coords, fits_file, i)
                self.image_ready.emit(pixmap)
                self.analyzer.save_detection_visual(data, coords, fits_file, i)
        
            # Use pre-calculated zenith angle
            zenith_angle = zenith_angles[i] if i < len(zenith_angles) else np.nan
        
            result = {
                'Time': timestamps[i].isot if i < len(timestamps) else '',
                'zenith': zenith_angle,
                'Ax': coords['A_up'][0], 'Ay': coords['A_up'][1],
                'Bx': coords['B_down'][0], 'By': coords['B_down'][1],
                'Cx': coords['C_left'][0], 'Cy': coords['C_left'][1],
                'Dx': coords['D_right'][0], 'Dy': coords['D_right'][1],
                'Img_X': coords['Star'][0], 'Img_Y': coords['Star'][1]
            }
            results.append(result)
    
        import pandas as pd
        df = pd.DataFrame(results)
    
        # Calculations
        df['C. Center (X)'] = (df['Ax'] + df['Bx'] + df['Cx'] + df['Dx']) / 4
        df['C. Center (Y)'] = (df['Ay'] + df['By'] + df['Cy'] + df['Dy']) / 4
        df['scale_factor'] = 16 / (df['By'] - df['Ay'])
        df['Correction_X'] = (df['Img_X'] - df['C. Center (X)']) * df['scale_factor'] * 0.318 * 60
        df['Correction_Y'] = (df['C. Center (Y)'] - df['Img_Y']) * df['scale_factor'] * 0.318 * 60
    
        return df
    
    def create_detection_pixmap(self, data, coords, fits_file, i):
        """Create QPixmap from detection image - ALL WHITE CIRCLES"""
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
        
            data_clean = np.nan_to_num(data, nan=0.0)
            if data_clean.max() > 0:
                vmin, vmax = np.percentile(data_clean[data_clean > data_clean.mean()], [5, 98])
                ax.imshow(data_clean, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        
            # Draw ALL circles in WHITE
            names = ['A_up', 'B_down', 'C_left', 'D_right', 'Star']
        
            for name in names:
                if name in coords and not np.isnan(coords[name][0]):
                    x, y = coords[name]
                    circle = plt.Circle((x, y), 10, color='white', fill=False, linewidth=2.5)
                    ax.add_patch(circle)
        
            ax.set_title(f'Detection #{i+1}: {fits_file}', fontsize=10)
            ax.axis('off')
            plt.tight_layout()
        
            # Convert to QPixmap
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
        
            img = Image.open(buf)
            img = img.convert('RGB')
        
            # Convert to QPixmap
            from PyQt5.QtGui import QImage
            data = img.tobytes('raw', 'RGB')
            qimage = QImage(data, img.width, img.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
            return pixmap
        
        except Exception as e:
            print(f"Image creation error: {e}")
            return None

    def find_star_wide_search(self, data, ref_x, ref_y, search_radius):
        """Wide search for star during transition zone"""
        try:
            x0, y0 = int(ref_x), int(ref_y)
        
            # Safe bounds
            x1 = max(0, x0 - search_radius)
            x2 = min(data.shape[1], x0 + search_radius)
            y1 = max(0, y0 - search_radius)
            y2 = min(data.shape[0], y0 + search_radius)
        
            cutout = data[y1:y2, x1:x2].copy()
        
            # Find brightest point
            peak_idx = np.unravel_index(np.argmax(cutout), cutout.shape)
            peak_val = cutout[peak_idx]
        
            # Verify it's a real star
            if peak_val > np.median(cutout) + 5 * np.std(cutout):
                return x1 + peak_idx[1], y1 + peak_idx[0]
        
            return None, None
        
        except Exception as e:
            return None, None


class ModernButton(QPushButton):
    """Custom styled button"""
    def __init__(self, text, icon=None, color="#6366f1"):
        super().__init__(text)
        self.default_color = color
        self.hover_color = self.lighten_color(color)
        self.setMinimumHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        
        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(24, 24))
        
        self.update_style(self.default_color)
        
    def lighten_color(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = min(255, r + 30)
        g = min(255, g + 30)
        b = min(255, b + 30)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def update_style(self, color):
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color}, stop:1 {self.lighten_color(color)});
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self.hover_color}, stop:1 {self.lighten_color(self.hover_color)});
            }}
            QPushButton:pressed {{
                background: {self.default_color};
            }}
            QPushButton:disabled {{
                background: #9ca3af;
            }}
        """)
    
    def enterEvent(self, event):
        self.update_style(self.hover_color)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.update_style(self.default_color)
        super().leaveEvent(event)


class TACTICMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = TACTICPointingCorrection()
        
        # Set default output_dir if not exists
        if not hasattr(self.analyzer, 'output_dir'):
            self.analyzer.output_dir = os.path.dirname(self.analyzer.fits_dir)
        
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("TACTIC Pointing Correction")
        # Get screen size
        screen = QApplication.desktop().screenGeometry()
        width = min(1100, screen.width() - 100)
        height = min(800, screen.height() - 100)
    
        self.setGeometry(50, 50, width, height)
        self.setMinimumSize(1050, 700)
        self.setWindowIcon(self.create_telescope_icon())
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e1b4b, stop:0.5 #581c87, stop:1 #1e1b4b);
            }
        """)
        
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        header = self.create_header()
        main_layout.addWidget(header)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                background: rgba(255, 255, 255, 0.05);
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                padding: 8px 20px;
                margin-right: 5px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #a855f7);
            }
        """)
        
        self.create_setup_tab()
        self.create_analysis_tab()
        self.create_results_tab()
        
        main_layout.addWidget(self.tabs)
        
    def create_header(self):
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f46e5, stop:1 #a855f7);
                border-radius: 5px;
                padding: 0px;
            }
        """)
        
        layout = QVBoxLayout(header)
        
        title = QLabel("TACTIC Pointing Correction")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Developed by Mahendra Kothari and Muskan Maheshwari")
        subtitle.setStyleSheet("font-size: 12px; color: rgba(255, 255, 255, 0.7); font-style: italic;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        return header
    
    def create_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Source Configuration
        source_group = self.create_group_box("🎯 Source Configuration")
        source_layout = QHBoxLayout()

        self.source_name = self.create_input("", self.analyzer.SOURCE_NAME)
        self.source_ra = self.create_input("", str(self.analyzer.SOURCE_RA))
        self.source_dec = self.create_input("", str(self.analyzer.SOURCE_DEC))

        source_layout.addWidget(QLabel("Source Name:"))
        source_layout.addWidget(self.source_name)
        source_layout.addWidget(QLabel("RA (deg):"))
        source_layout.addWidget(self.source_ra)
        source_layout.addWidget(QLabel("DEC (deg):"))
        source_layout.addWidget(self.source_dec)
    
        source_group.setLayout(source_layout)
        source_group.setMaximumHeight(90)
        scroll_layout.addWidget(source_group)
        
        # Observatory Configuration
        obs_group = self.create_group_box("🌍 Observatory Configuration")
        obs_layout = QHBoxLayout()
    
        self.obs_lat = self.create_input("", str(self.analyzer.OBSERVATORY_LAT))
        self.obs_lon = self.create_input("", str(self.analyzer.OBSERVATORY_LON))
        self.obs_height = self.create_input("", str(self.analyzer.OBSERVATORY_HEIGHT))

        obs_layout.addWidget(QLabel("Latitude:"))
        obs_layout.addWidget(self.obs_lat)
        obs_layout.addWidget(QLabel("Longitude:"))
        obs_layout.addWidget(self.obs_lon)
        obs_layout.addWidget(QLabel("Height (m):"))
        obs_layout.addWidget(self.obs_height)

        obs_group.setLayout(obs_layout)
        obs_group.setMaximumHeight(90)
        scroll_layout.addWidget(obs_group)
        
        # File Selection
        file_group = self.create_group_box("📁 File Selection")
        file_layout = QVBoxLayout()
        
        fits_layout = QHBoxLayout()
        self.fits_dir_label = QLabel(self.analyzer.fits_dir)
        self.fits_dir_label.setStyleSheet("color: #4ade80; padding: 8px;")
        fits_btn = ModernButton("Select FITS Directory", color="#10b981")
        fits_btn.clicked.connect(self.select_fits_dir)
        fits_layout.addWidget(QLabel("FITS Directory:"))
        fits_layout.addWidget(self.fits_dir_label, 1)
        fits_layout.addWidget(fits_btn)
        
        ts_layout = QHBoxLayout()
        self.ts_file_label = QLabel(self.analyzer.timestamp_file)
        self.ts_file_label.setStyleSheet("color: #4ade80; padding: 8px;")
        ts_btn = ModernButton("Select Timestamp File", color="#3b82f6")
        ts_btn.clicked.connect(self.select_timestamp_file)
        ts_layout.addWidget(QLabel("Timestamp File:"))
        ts_layout.addWidget(self.ts_file_label, 1)
        ts_layout.addWidget(ts_btn)
        
        output_layout = QHBoxLayout()
        self.output_dir_label = QLabel(self.analyzer.output_dir)
        self.output_dir_label.setStyleSheet("color: #4ade80; padding: 8px;")
        output_btn = ModernButton("Select Output Directory", color="#f59e0b")
        output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addWidget(self.output_dir_label, 1)
        output_layout.addWidget(output_btn)
        
        file_layout.addLayout(fits_layout)
        file_layout.addLayout(ts_layout)
        file_layout.addLayout(output_layout)
        
        self.save_all_cb = QCheckBox("✓ Save ALL verification images (default: first 5)")
        self.save_all_cb.setStyleSheet("""
            QCheckBox {
                color: #fbbf24;
                font-size: 15px;
                font-weight: bold;
                padding: 15px;
                background: rgba(251, 191, 36, 0.1);
                border: 2px solid rgba(251, 191, 36, 0.3);
                border-radius: 10px;
            }
            QCheckBox::indicator { 
                width: 28px; 
                height: 28px;
                border: 3px solid #fbbf24;
                border-radius: 6px;
                background: rgba(0, 0, 0, 0.5);
            }
            QCheckBox::indicator:checked {
                background: #fbbf24;
            }
        """)

        file_layout.addWidget(self.save_all_cb)
        
        file_group.setLayout(file_layout)
        file_group.setMaximumHeight(235)
        scroll_layout.addWidget(file_group)
        
        # Reference Points - CORRECTED VERSION
        ref_group = self.create_group_box("🎯 Reference Points (LED & Star Coordinates)")
        ref_layout = QVBoxLayout()
        
        # Info with hint
        info_layout = QVBoxLayout()
        info_label = QLabel("Enter LED coordinates (same for all) and Star coordinates:")
        info_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 11px; padding: 5px;")
        info_layout.addWidget(info_label)
        
        hint_label = QLabel("💡 Tip: Enter available star coords (Pre/Post) or set unused to 0,0 for auto-detect")
        hint_label.setStyleSheet("""
            color: #fbbf24;
            font-size: 10px;
            font-style: italic;
            padding: 5px;
            background: rgba(251, 191, 36, 0.1);
            border-radius: 5px;
        """)
        info_layout.addWidget(hint_label)
        ref_layout.addLayout(info_layout)
        
     
        # Grid for coordinates - COMPACT
        ref_grid = QGridLayout()
                 
        ref_grid.setContentsMargins(0, 0, 0, 0)

        # Column widths control
        ref_grid.setColumnMinimumWidth(0, 80)  # "LED A (Top):"
        ref_grid.setColumnMinimumWidth(1, 4)   # "X:"
        ref_grid.setColumnMinimumWidth(2, 80)   # X box
        ref_grid.setColumnMinimumWidth(3, 18)   # "Y:"
        ref_grid.setColumnMinimumWidth(4, 80)   # Y box

        # Prevent auto-stretch
        for c in range(5):
            ref_grid.setColumnStretch(c, 0)

        ref_grid.setColumnStretch(0, 0)  # Label column - no stretch
        ref_grid.setColumnStretch(1, 0)  # X label - no stretch
        ref_grid.setColumnStretch(2, 0)  # X input - no stretch
        ref_grid.setColumnStretch(3, 0)  # Y label - no stretch
        ref_grid.setColumnStretch(4, 0)  # Y input - no stretch
        
        self.ref_inputs = {}
        led_names = [
            ('A_up', 'LED A (Top)', 0),
            ('B_down', 'LED B (Bottom)', 1),
            ('C_left', 'LED C (Left)', 2),
            ('D_right', 'LED D (Right)', 3),
            ('Star_Pre', 'Star (Pre-Transit)', 4),
            ('Star_Post', 'Star (Post-Transit)', 5)
        ]
        
        for key, label, row in led_names:

            # -------- LED label --------
            lbl = QLabel(label + ":")
            lbl.setStyleSheet("color: white; font-weight: bold;")
            lbl.setFixedWidth(110)   # LED text ke baad ka GAP yahin control hoga
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            # -------- default values --------
            if key == 'Star_Pre':
                default_x = self.analyzer.reference_points.get('Star', (555.5624, 334.4543))[0]
                default_y = self.analyzer.reference_points.get('Star', (555.5624, 334.4543))[1]
            elif key == 'Star_Post':
                default_x = 0
                default_y = 0
            elif key in self.analyzer.reference_points:
                default_x = self.analyzer.reference_points[key][0]
                default_y = self.analyzer.reference_points[key][1]
            else:
                default_x = 0
                default_y = 0

            # -------- inputs --------
            x_input = self.create_input("X", str(default_x))
            x_input.setPlaceholderText("X coordinate")
            x_input.setFixedWidth(95)
            x_input.setMaximumHeight(30)
        
            y_input = self.create_input("Y", str(default_y))
            y_input.setPlaceholderText("Y coordinate")
            y_input.setFixedWidth(95)
            y_input.setMaximumHeight(30)
        
            # -------- row layout (THIS removes GAP1) --------
            row_layout = QHBoxLayout()
            row_layout.setSpacing(4)
            row_layout.setContentsMargins(0, 0, 0, 0)

            row_layout.addWidget(lbl)

            x_lbl = QLabel("X:")
            x_lbl.setFixedWidth(14)
            row_layout.addWidget(x_lbl)
            row_layout.addWidget(x_input)

            y_lbl = QLabel("Y:")
            y_lbl.setFixedWidth(14)
            row_layout.addWidget(y_lbl)
            row_layout.addWidget(y_input)

            row_layout.addStretch()

            # -------- put whole row into grid --------
            ref_grid.addLayout(row_layout, row, 0, 1, 5)

            self.ref_inputs[key] = {'x': x_input, 'y': y_input}

        ref_grid.setHorizontalSpacing(2)
        ref_layout.addLayout(ref_grid)
        
        # Buttons
        ref_buttons_layout = QHBoxLayout()
        
        load_ref_btn = ModernButton("📂 Load from File", color="#8b5cf6")
        load_ref_btn.setMaximumWidth(160)
        load_ref_btn.clicked.connect(self.load_reference_points)
        ref_buttons_layout.addWidget(load_ref_btn)
        
        save_ref_btn = ModernButton("💾 Save to File", color="#ec4899")
        save_ref_btn.setMaximumWidth(160)
        save_ref_btn.clicked.connect(self.save_reference_points)
        ref_buttons_layout.addWidget(save_ref_btn)
        
        reset_ref_btn = ModernButton("🔄 Reset to Default", color="#6366f1")
        reset_ref_btn.setMaximumWidth(160)
        reset_ref_btn.clicked.connect(self.reset_reference_points)
        ref_buttons_layout.addWidget(reset_ref_btn)
        
        ref_layout.addLayout(ref_buttons_layout)
        ref_group.setLayout(ref_layout)
        #ref_group.setMaximumHeight(320)
        ref_group.setMaximumWidth(600)                      

        # === Polynomial Fitting Order Selection ===
        poly_group = QGroupBox("Polynomial Fitting Options")
        poly_group.setMaximumWidth(300)
        poly_group.setStyleSheet("""
            QGroupBox {
                background: rgba(60, 60, 80, 0.5);
                border: 2px solid rgba(100, 150, 255, 0.3);
                border-radius: 10px;
                margin-top: 10px;
                padding: 8px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
        poly_main_layout = QVBoxLayout()
        poly_main_layout.setSpacing(5)  # Compact spacing
        poly_main_layout.setContentsMargins(5, 5, 5, 5)  # Margins kam
        poly_group.setLayout(poly_main_layout)
        poly_group.setMaximumHeight(180)  # Height limit

        # Create horizontal layout for Reference Points + Polynomial Options
        ref_poly_layout = QHBoxLayout()
        ref_poly_layout.setSpacing(15)
        ref_poly_layout.setContentsMargins(0, 0, 0, 0)

        ref_poly_layout.addWidget(ref_group)
        ref_poly_layout.addWidget(poly_group)

        #  MINIMUM stretch zaroori hai, warna block hide ho jata hai
        ref_poly_layout.setStretch(0, 1)
        ref_poly_layout.setStretch(1, 0)
        
        scroll_layout.addLayout(ref_poly_layout)

        
        # PRE-TRANSIT Section
        pre_label = QLabel("PRE-Transit Fitting Orders:")
        pre_label.setStyleSheet("color: #4ecdc4; font-size: 13px; font-weight: bold; margin-top: 5px;")
        poly_main_layout.addWidget(pre_label)
        
        pre_check_layout = QHBoxLayout()
        self.pre_check1 = QCheckBox("1st")
        self.pre_check2 = QCheckBox("2nd")
        self.pre_check3 = QCheckBox("3rd")
        self.pre_check4 = QCheckBox("4th")
        self.pre_check5 = QCheckBox("5th")

        # Default: 1,2,3 checked
        self.pre_check1.setChecked(True)
        self.pre_check2.setChecked(True)
        self.pre_check3.setChecked(True)

        for check in [self.pre_check1, self.pre_check2, self.pre_check3, self.pre_check4, self.pre_check5]:
            check.setStyleSheet("""
                QCheckBox {
                    color: white;
                    font-size: 11px;
                    spacing: 5px;
                }
                QCheckBox::indicator {
                    width: 15px;
                    height: 15px;
                    border: 2px solid rgba(100, 150, 255, 0.5);
                    border-radius: 4px;
                    background: rgba(40, 40, 60, 0.8);
                }
                QCheckBox::indicator:checked {
                    background: #4ecdc4;
                    border: 2px solid #4ecdc4;
                }
            """)
            pre_check_layout.addWidget(check)
        
        pre_check_layout.addStretch()
        poly_main_layout.addLayout(pre_check_layout)
        
        # POST-TRANSIT Section
        post_label = QLabel("POST-Transit Fitting Orders:")
        post_label.setStyleSheet("color: #ff6b6b; font-size: 13px; font-weight: bold; margin-top: 10px;")
        poly_main_layout.addWidget(post_label)

        post_check_layout = QHBoxLayout()
        self.post_check1 = QCheckBox("1st")
        self.post_check2 = QCheckBox("2nd")
        self.post_check3 = QCheckBox("3rd")
        self.post_check4 = QCheckBox("4th")
        self.post_check5 = QCheckBox("5th")

        # Default: 1,2,3 checked
        self.post_check1.setChecked(True)
        self.post_check2.setChecked(True)
        self.post_check3.setChecked(True)

        for check in [self.post_check1, self.post_check2, self.post_check3, self.post_check4, self.post_check5]:
            check.setStyleSheet("""
                QCheckBox {
                    color: white;
                    font-size: 12px;
                    spacing: 8px;
                }
                QCheckBox::indicator {
                    width: 15px;
                    height: 15px;
                    border: 2px solid rgba(100, 150, 255, 0.5);
                    border-radius: 4px;
                    background: rgba(40, 40, 60, 0.8);
                }
                QCheckBox::indicator:checked {
                    background: #ff6b6b;
                    border: 2px solid #ff6b6b;
                }
            """)
            post_check_layout.addWidget(check)

        post_check_layout.addStretch()
        poly_main_layout.addLayout(post_check_layout)
                
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        apply_btn = ModernButton("💾 Apply Settings", color="#8b5cf6")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn)
        
        self.tabs.addTab(tab, "⚙  Setup")
    
    def create_analysis_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        
        # Control buttons
        control_group = self.create_group_box("🚀 Analysis Control")
        control_layout = QHBoxLayout()
    
        self.start_btn = ModernButton("▶️ START", color="#10b981")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.clicked.connect(self.start_analysis)
    
        self.pause_btn = ModernButton("⏸️ PAUSE", color="#f59e0b")
        self.pause_btn.setMinimumHeight(50)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_analysis)
    
        self.status_label = QLabel("Ready to start")
        self.status_label.setStyleSheet("""
            QLabel {
                background: rgba(100, 100, 100, 0.3);
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
    
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                text-align: center;
                background: rgba(0, 0, 0, 0.3);
                color: white;
                font-weight: bold;
                min-height: 50px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:0.5 #3b82f6, stop:1 #a855f7);
                border-radius: 8px;
            }
        """)
        self.progress_bar.setValue(0)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.progress_bar)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
    
        
        
        # Bottom section
        bottom_layout = QHBoxLayout()

        # Live View
        live_group = self.create_group_box("🖼️ Live Detection View")
        live_layout = QVBoxLayout()

        self.image_label = QLabel("Processing...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(450, 450)
        self.image_label.setMaximumSize(600, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                color: rgba(255, 255, 255, 0.5);
                font-size: 12px;
            }
        """)

        live_layout.addWidget(self.image_label)
        live_group.setLayout(live_layout)
        bottom_layout.addWidget(live_group, 2)
        # Activity Log (Right side)
        log_group = self.create_group_box("📝 Activity Log")
        log_layout = QVBoxLayout()
    
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(500)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 0, 0, 0.4);
                color: #4ade80;
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 10px;
                font-family: 'Courier New';
                font-size: 11px;
            }
        """)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        bottom_layout.addWidget(log_group, 2)
        bottom_layout.setSpacing(5)
                        
        main_layout.addLayout(bottom_layout)
        
        self.tabs.addTab(tab, "🔬  Analyze")
    
    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
    
        self.indicators = {}  # Keep for compatibility            
        # Download Results
        results_group = self.create_group_box("📥 Download Results")
        results_layout = QVBoxLayout()
        
        # Horizontal layout for all download buttons in one line
        downloads_h_layout = QHBoxLayout()
        downloads_h_layout.setSpacing(8)

        downloads = [
            ("📊 Excel", "pointing_correction_results.xlsx", "#10b981"),
            ("📈 Plot", "pointing_fits_pre_post.png", "#3b82f6"),
            ("📄 TXT", "fit_results_pre_post.txt", "#a855f7"),
            ("🖼️ Images", "detection_images", "#f59e0b"),
        ]

        for title, filename, color in downloads:
            btn = ModernButton(f"{title}", color=color)
            btn.setMaximumHeight(45)
            btn.setMinimumWidth(100)
            btn.setMaximumWidth(150)
            
            btn.clicked.connect(partial(self.open_result_file, filename))
            downloads_h_layout.addWidget(btn)
        downloads_h_layout.addStretch()
        results_layout.addLayout(downloads_h_layout)
        
        results_layout.addStretch()
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Plot preview
        plot_group = self.create_group_box("📊 Pre/Post Transit Plot Preview")
        plot_layout = QVBoxLayout()

        self.plot_preview_label = QLabel("Run analysis to see plot")
        self.plot_preview_label.setAlignment(Qt.AlignCenter)
        self.plot_preview_label.setMinimumSize(700, 600)
        self.plot_preview_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                color: rgba(255, 255, 255, 0.5);
            }
        """)
        self.plot_preview_label.setScaledContents(True)

        plot_layout.addWidget(self.plot_preview_label)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        self.tabs.addTab(tab, "💾  Results")
        
    def load_plot_preview(self):
        """Load plot preview in Results tab"""
        plot_path = os.path.join(self.analyzer.output_dir, 'pointing_fits_pre_post.png')
        if os.path.exists(plot_path):
            pixmap = QPixmap(plot_path)
            scaled = pixmap.scaled(self.plot_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.plot_preview_label.setPixmap(scaled)
    
    def create_group_box(self, title):
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                margin-top: 15px;
                padding-top: 15px;
                font-size: 14px;
                font-weight: bold;
                background: rgba(255, 255, 255, 0.05);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
            }
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 12px;
            }
        """)
        return group
    
    def create_input(self, label, default_value):
        input_field = QLineEdit(default_value)
        input_field.setMinimumHeight(35)
        input_field.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 10px;
                color: white;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 2px solid #6366f1;
            }
        """)
        return input_field
    
    def create_telescope_icon(self):
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QPen(QColor("#6366f1"), 4))
        painter.setBrush(QColor("#a855f7"))
        painter.drawEllipse(10, 10, 44, 44)
        painter.drawLine(32, 32, 50, 14)
        
        painter.end()
        return QIcon(pixmap)
    
    def select_fits_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select FITS Directory")
        if dir_path:
            self.fits_dir_label.setText(dir_path)
            self.analyzer.fits_dir = dir_path
            self.add_log(f"FITS directory: {dir_path}", "success")
    
    def select_timestamp_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Timestamp File", "", "Text Files (*.txt)")
        if file_path:
            self.ts_file_label.setText(file_path)
            self.analyzer.timestamp_file = file_path
            self.add_log(f"Timestamp file: {file_path}", "success")
    
    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_label.setText(dir_path)
            self.analyzer.output_dir = dir_path
            self.add_log(f"Output directory: {dir_path}", "success")
    
    def load_reference_points(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference Points", "", 
            "Text Files (*.txt);;All Files (*.*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 3:
                        led_name = parts[0]
                        x_coord = float(parts[1])
                        y_coord = float(parts[2])
                        
                        # Handle both old 'Star' and new 'Star_Pre' format
                        if led_name == 'Star' and 'Star' not in self.ref_inputs:
                            led_name = 'Star_Pre'
                        
                        if led_name in self.ref_inputs:
                            self.ref_inputs[led_name]['x'].setText(str(x_coord))
                            self.ref_inputs[led_name]['y'].setText(str(y_coord))
                
                QMessageBox.information(self, "Success", f"✅ Reference points loaded from:\n{file_path}")
                self.add_log(f"Reference points loaded from: {file_path}", "success")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"❌ Failed to load file:\n{str(e)}")
    
    def save_reference_points(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Reference Points", "reference_points.txt",
            "Text Files (*.txt);;All Files (*.*)"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("# TACTIC Reference Points\n")
                    f.write("# Format: LED_NAME X_coordinate Y_coordinate\n")
                    f.write("# Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
                    
                    for key in ['A_up', 'B_down', 'C_left', 'D_right', 'Star_Pre', 'Star_Post']:
                        if key in self.ref_inputs:
                            x_val = self.ref_inputs[key]['x'].text()
                            y_val = self.ref_inputs[key]['y'].text()
                            f.write(f"{key} {x_val} {y_val}\n")
                
                QMessageBox.information(self, "Success", f"✅ Reference points saved to:\n{file_path}")
                self.add_log(f"Reference points saved to: {file_path}", "success")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"❌ Failed to save file:\n{str(e)}")
    
    def reset_reference_points(self):
        default_refs = {
            'A_up': (544.5412, 556.8081),
            'B_down': (542.0328, 121.6288),
            'C_left': (332.5018, 335.6305),
            'D_right': (755.7680, 335.9140),
            'Star_Pre': (555.5624, 334.4543),
            'Star_Post': (0, 0)
        }
        
        for key, (x, y) in default_refs.items():
            if key in self.ref_inputs:
                self.ref_inputs[key]['x'].setText(str(x))
                self.ref_inputs[key]['y'].setText(str(y))
        
        QMessageBox.information(self, "Reset", "✅ Reference points reset to default values")
        self.add_log("Reference points reset to defaults", "info")
    
    def apply_settings(self):
        try:
            self.analyzer.SOURCE_NAME = self.source_name.text()
            # Handle save_all_cb checkbox
            if self.save_all_cb.isChecked():
                self.analyzer.save_count = 999999  # Save all images
            else:
                self.analyzer.save_count = 5  # Default 5 images
            self.analyzer.SOURCE_RA = float(self.source_ra.text())
            self.analyzer.SOURCE_DEC = float(self.source_dec.text())
            self.analyzer.OBSERVATORY_LAT = float(self.obs_lat.text())
            self.analyzer.OBSERVATORY_LON = float(self.obs_lon.text())
            self.analyzer.OBSERVATORY_HEIGHT = float(self.obs_height.text())
            
            # RECREATE location
            from astropy.coordinates import EarthLocation
            import astropy.units as u
            
            self.analyzer.location = EarthLocation(
                lat=self.analyzer.OBSERVATORY_LAT * u.deg, 
                lon=self.analyzer.OBSERVATORY_LON * u.deg,
                height=self.analyzer.OBSERVATORY_HEIGHT * u.m
            )
            
            # RECREATE target_coord
            from astropy.coordinates import SkyCoord
            
            self.analyzer.target_coord = SkyCoord(
                ra=self.analyzer.SOURCE_RA * u.deg,
                dec=self.analyzer.SOURCE_DEC * u.deg,
                frame='icrs'
            )
            
            # VERIFICATION PRINT
            print(f"\n✓ Applied Settings:")
            print(f"  Source: {self.analyzer.SOURCE_NAME}")
            print(f"  RA: {self.analyzer.SOURCE_RA:.6f}° ({self.analyzer.SOURCE_RA/15:.6f}h)")
            print(f"  DEC: {self.analyzer.SOURCE_DEC:.6f}°")
            print(f"  Target: {self.analyzer.target_coord}\n")
            
            # Update LED coordinates
            for key in ['A_up', 'B_down', 'C_left', 'D_right']:
                x_val = float(self.ref_inputs[key]['x'].text())
                y_val = float(self.ref_inputs[key]['y'].text())
                self.analyzer.reference_points[key] = (x_val, y_val)

            # Handle Star coordinates separately
            star_pre_x = float(self.ref_inputs['Star_Pre']['x'].text())
            star_pre_y = float(self.ref_inputs['Star_Pre']['y'].text())
            star_post_x = float(self.ref_inputs['Star_Post']['x'].text())
            star_post_y = float(self.ref_inputs['Star_Post']['y'].text())

            # Store both
            self.analyzer.reference_points['Star_Pre'] = (star_pre_x, star_pre_y)
            self.analyzer.reference_points['Star_Post'] = (star_post_x, star_post_y)

            # For backward compatibility
            self.analyzer.reference_points['Star'] = (star_pre_x, star_pre_y)
            
            QMessageBox.information(self, "Success", "✅ All settings applied successfully!")
            self.add_log("Settings applied successfully", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Invalid input: {str(e)}")
            self.add_log(f"Settings error: {str(e)}", "error")
    
    def start_analysis(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Running", "Analysis is already running!")
            return
    
        # Get PRE-transit orders from checkboxes
        pre_orders = []
        if self.pre_check1.isChecked():
            pre_orders.append(1)
        if self.pre_check2.isChecked():
            pre_orders.append(2)
        if self.pre_check3.isChecked():
            pre_orders.append(3)
        if self.pre_check4.isChecked():
            pre_orders.append(4)
        if self.pre_check5.isChecked():
            pre_orders.append(5)
    
        # Get POST-transit orders from checkboxes
        post_orders = []
        if self.post_check1.isChecked():
            post_orders.append(1)
        if self.post_check2.isChecked():
            post_orders.append(2)
        if self.post_check3.isChecked():
            post_orders.append(3)
        if self.post_check4.isChecked():
            post_orders.append(4)
        if self.post_check5.isChecked():
            post_orders.append(5)
    
        # Validation
        if not pre_orders:
            pre_orders = [1, 2, 3]
            QMessageBox.warning(self, "Warning", "No PRE-transit orders selected. Using default: 1,2,3")
    
        if not post_orders:
            post_orders = [1, 2, 3]
            QMessageBox.warning(self, "Warning", "No POST-transit orders selected. Using default: 1,2,3")
    
        self.add_log(f"PRE-Transit orders: {pre_orders}", "info")
        self.add_log(f"POST-Transit orders: {post_orders}", "info")
    
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.start_btn.setText("⏳ Running...")
        self.status_label.setText("Starting...")
        self.log_text.clear()
    
        self.worker = WorkerThread(self.analyzer, pre_orders=pre_orders, post_orders=post_orders)
        self.worker.progress.connect(self.update_progress)
        self.worker.log_message.connect(self.add_log)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.image_ready.connect(self.update_image)
        self.worker.indicator_update.connect(self.update_indicator)
        self.worker.start()
        
    def pause_analysis(self):
        """Pause/Resume the analysis"""
        if not self.worker or not self.worker.isRunning():
            return
    
        if self.worker.is_paused():
            self.worker.resume()
            self.pause_btn.setText("⏸️ PAUSE")
            self.status_label.setText("Running...")
        else:
            self.worker.pause()
            self.pause_btn.setText("▶️ RESUME")
            self.status_label.setText("Paused")
   
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Running... {value}%")
    
    def add_log(self, message, msg_type):
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {"success": "#4ade80", "error": "#ef4444", "info": "#60a5fa"}[msg_type]
        self.log_text.append(f'<span style="color: {color};">[{timestamp}] {message}</span>')
        
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_image(self, pixmap):
        if pixmap:
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
    
    def update_indicator(self, indicator_name, active):
        if indicator_name in self.indicators:
            indicator = self.indicators[indicator_name]
            
            if 'button' in indicator:
                if active:
                    indicator['label'].setText("🟢")
                    indicator['button'].setStyleSheet(f"""
                        QPushButton {{
                            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 {indicator['color']}, stop:1 rgba(0,0,0,0.5));
                            color: white;
                            border: 2px solid {indicator['color']};
                            border-radius: 12px;
                            font-size: 13px;
                            font-weight: bold;
                            padding: 10px 20px;
                        }}
                    """)
                else:
                    indicator['label'].setText("⚪")
                    indicator['button'].setStyleSheet("""
                        QPushButton {
                            background: rgba(100, 100, 100, 0.3);
                            color: rgba(255, 255, 255, 0.5);
                            border: 2px solid rgba(255,255, 255, 0.1);
                            border-radius: 12px;
                            font-size: 13px;
                            padding: 10px 20px;
                        }
                    """)
    
    def analysis_finished(self):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶️ START")
        self.pause_btn.setEnabled(False)
        self.status_label.setText("Complete!")  
        self.load_plot_preview()
        QMessageBox.information(
            self, "Complete", 
            f"✅ Analysis completed successfully!\n\nResults saved in:\n{self.analyzer.output_dir}\n\nCheck Results tab for downloads."
        )
    
    def open_result_file(self, filename):
        full_path = os.path.join(self.analyzer.output_dir, filename)
        
        if not os.path.exists(full_path):
            QMessageBox.warning(
                self, "File Not Found", 
                f"File not found:\n{full_path}\n\nPlease run analysis first to generate results."
            )
            return
        
        try:
            if sys.platform == "win32":
                os.startfile(full_path)
            elif sys.platform == "darwin":
                os.system(f'open "{full_path}"')
            else:
                os.system(f'xdg-open "{full_path}"')
            
            self.add_log(f"Opened: {filename}", "success")
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to open file:\n{full_path}\n\nError: {str(e)}"
            )
            self.add_log(f"Failed to open {filename}: {str(e)}", "error")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = TACTICMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
