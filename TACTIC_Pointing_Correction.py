import os
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.time import Time
from photutils.detection import DAOStarFinder
from photutils.background import MedianBackground
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TACTICPointingCorrection:
    def __init__(self):
        self.OBSERVATORY_LAT = 24.6548
        self.OBSERVATORY_LON = 72.7792
        self.OBSERVATORY_HEIGHT = 1360
        self.SOURCE_RA = 5.627411 * 15
        self.SOURCE_DEC = 21.142584
        self.SOURCE_NAME = "MU_Geminorum"  # Edit
        self.IST_UTC_OFFSET = 5.5  # hours (IST = UTC + 5:30)

        # Reference points - UPDATE THESE WITH YOUR ACTUAL VALUES
        self.reference_points = {
            'A_up': (546.0919, 546.2835),    # 1024-467.1919
            'B_down': (543.5715, 119.8698),  # 1024-902.3712
            'C_left': (332.5692, 336.5287),  # 1024-688.3695
            'D_right': (755.4114, 335.6598),  # 1024-688.0860
            'Star': (554.2152, 301.6761)     # 1024-689.5457
        }

        self.fits_dir = "/home/goals/Pointing2025/MU_Geminorum_09Jan2026"
        self.timestamp_file = "/home/goals/Pointing2025/MU_Geminorum_09Jan2026/timestemp.txt"
        self.output_xls = "pointing_correction_results.xls"
        self.output_dir = "/home/goals/Pointing2025/MU_Geminorum_09Jan2026/Output1"
        self.save_count = 5  # Default value
        from astropy.coordinates import EarthLocation, SkyCoord, AltAz
        import astropy.units as u

        # Observatory location
        self.location = EarthLocation(
            lat=self.OBSERVATORY_LAT * u.deg,
            lon=self.OBSERVATORY_LON * u.deg,
            height=self.OBSERVATORY_HEIGHT * u.m
        )

        # Target coordinates (J2000 frame)
        self.target_coord = SkyCoord(
            ra=self.SOURCE_RA * u.deg,
            dec=self.SOURCE_DEC * u.deg,
            frame='icrs'  # Add explicit frame
        )

    def parse_timestamps(self):
        """
        Parse timestamp file and convert IST to UTC for accurate zenith calculations.

        File format:
        Line 1: Date (YYYY-MM-DD or YYYY_MM-DD)
        Lines 2+: Times in IST (HH:MM)

        Returns:
        --------
        list of astropy.time.Time objects in UTC
        """
        from astropy.time import Time
        from datetime import datetime, timedelta

        print(f"Reading timestamps from: {self.timestamp_file}")

        timestamps = []

        with open(self.timestamp_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("Timestamp file is empty!")

        # Parse date (first line)
        date_str = lines[0].strip()
        # Handle both separator formats
        date_str = date_str.replace('_', '-')

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            print(f"Date: {date_obj.date()}")
        except ValueError as e:
            raise ValueError(f"Invalid date format in first line: {date_str}\n"
                             f"Expected: YYYY-MM-DD or YYYY_MM-DD\n"
                             f"Error: {e}")

        # Parse time entries (lines 2+)
        ist_count = 0
        utc_count = 0

        for line_num, line in enumerate(lines[1:], start=2):
            time_str = line.strip()

            if not time_str:  # Skip empty lines
                continue

            try:
                # Parse IST time
                time_obj = datetime.strptime(time_str, "%H:%M")
                ist_count += 1

                # Create full IST datetime
                ist_datetime = datetime(
                    date_obj.year, date_obj.month, date_obj.day,
                    time_obj.hour, time_obj.minute, 0
                )

                # Convert IST to UTC (IST = UTC + 5:30, so UTC = IST - 5:30)
                utc_datetime = ist_datetime - \
                    timedelta(hours=self.IST_UTC_OFFSET)
                utc_count += 1

                # Create astropy Time object from UTC
                timestamps.append(Time(utc_datetime))

                # Debug for first few entries
                if ist_count <= 3:
                    print(
                        f"  {ist_count}. IST {time_str} → UTC {utc_datetime.strftime('%H:%M:%S')}")

            except ValueError as e:
                print(
                    f"Warning: Skipping invalid time at line {line_num}: '{time_str}' ({e})")
                continue

        print(f"✓ Parsed {ist_count} IST times → {utc_count} UTC timestamps")

        if not timestamps:
            raise ValueError("No valid timestamps found in file!")

        return timestamps

        print(f"Parsed {len(timestamps)} timestamps")
        if len(timestamps) > 0:
            print(f"First timestamp (UTC): {timestamps[0].iso}")

        return timestamps

    def find_led_star_advanced(self, data, ref_x, ref_y):
        """ULTRA ROBUST - NO ERRORS + SUBPIXEL ACCURACY"""
        roi_size = 50
        half = roi_size // 2

        # Safe bounds
        y1 = max(0, int(ref_y - half))
        y2 = min(data.shape[0], int(ref_y + half))
        x1 = max(0, int(ref_x - half))
        x2 = min(data.shape[1], int(ref_x + half))

        roi = data[y1:y2, x1:x2].astype(float)
        roi = np.nan_to_num(roi, nan=0.0)

        # Simple background subtraction (median)
        bkg_median = np.median(roi[roi > 0])
        roi_clean = roi - bkg_median

        # Find brightest region (5x5 box)
        yy, xx = np.unravel_index(np.argmax(roi_clean), roi_clean.shape)

        # Sub-pixel refinement (weighted centroid 5x5)
        y_start = max(0, yy - 2)
        y_end = min(roi_clean.shape[0], yy + 3)
        x_start = max(0, xx - 2)
        x_end = min(roi_clean.shape[1], xx + 3)

        local = roi_clean[y_start:y_end, x_start:x_end]
        yw, xw = np.ogrid[:local.shape[0], :local.shape[1]]
        total = np.sum(local)
        if total > 0:
            x_cent = (np.sum(local * xw) / total) + x_start
            y_cent = (np.sum(local * yw) / total) + y_start
        else:
            x_cent, y_cent = xx + 0.5, yy + 0.5

        return x1 + x_cent, y1 + y_cent

    def save_detection_visual(self, data, coords, fits_file, i):
        """CLEAN CIRCLES ONLY - No labels, no fill, just verification"""
        import matplotlib
        matplotlib.use('Agg')  # Non-GUI backend
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 12))

        # Perfect stretch for TACTIC LEDs/Star
        data_clean = np.nan_to_num(data, nan=0.0)
        # Check if data has valid pixels
        if data_clean.max() == 0:
            print(f"Warning: Image {fits_file} appears to be blank")
            plt.close()
            return
        vmin, vmax = np.percentile(
            data_clean[data_clean > data_clean.mean()], [5, 98])

        ax.imshow(
            data_clean,
            cmap='inferno',
            origin='lower',
            vmin=vmin,
            vmax=vmax)

        # ONLY THIN WHITE CIRCLES around detected positions
        names = ['A_up', 'B_down', 'C_left', 'D_right', 'Star']
        for name in names:
            if not np.isnan(coords[name][0]):
                x, y = coords[name]
                # SINGLE THIN WHITE CIRCLE (just bahar covering full LED/Star)
                circle = plt.Circle(
                    (x, y), 10, color='white', fill=False, linewidth=2.5)
                ax.add_patch(circle)

        ax.set_title(
            f'LEDs/Star Detection Verification - {fits_file}',
            fontsize=14)
        ax.axis('off')
        plt.tight_layout()

        # Use output_dir
        output_dir = os.path.join(self.output_dir, "detection_images")
        os.makedirs(output_dir, exist_ok=True)

        output_path = f'{output_dir}/detect_{i:03d}_{fits_file[:-4]}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)  # IMPORTANT - Close figure to free memory

        print(f"Saved: {output_path}")

    def process_fits_files(self, timestamps):
        results = []
        fits_files = sorted([f for f in os.listdir(
            self.fits_dir) if f.endswith('.fit')])

        print(f"Processing {len(fits_files)} FITS files...")

        # CHECK if save_count exists, otherwise set default
        if not hasattr(self, 'save_count'):
            self.save_count = 5  # Default to 5 images

        for i, fits_file in enumerate(fits_files):
            if i % 10 == 0:
                print(f"Processed {i}/{len(fits_files)}")

            fits_path = os.path.join(self.fits_dir, fits_file)
            try:
                with fits.open(fits_path) as hdul:
                    data = hdul[0].data.astype(float)
            except Exception as e:
                print(f"Skip {fits_file}: {e}")
                continue

            coords = {}
            for name, (ref_x, ref_y) in self.reference_points.items():
                x_found, y_found = self.find_led_star_advanced(
                    data, ref_x, ref_y)
                coords[name] = (x_found if x_found is not None else np.nan,
                                y_found if y_found is not None else np.nan)

            # Save detection images
            if i < self.save_count:
                try:
                    self.save_detection_visual(data, coords, fits_file, i)
                    print(f"Saved detection image {i+1}/{self.save_count}")
                except Exception as e:
                    print(f"Warning: Could not save detection image {i}: {e}")

            # Calculate zenith angle
            zenith_angle = np.nan
            if i < len(timestamps) and not np.isnan(coords['Star'][0]):
                try:
                    obstime = timestamps[i]

                    # Transform to AltAz frame
                    altaz = self.target_coord.transform_to(
                        AltAz(obstime=obstime, location=self.location)
                    )

                    # Zenith angle = 90° - altitude
                    zenith_angle = 90.0 - altaz.alt.deg

                    # Debug: print first few calculations
                    if i < 5:
                        print(f"\nDebug Zenith Calculation #{i}:")
                        print(f"  Time (UTC): {obstime.iso}")
                        print(
                            f"  Source: RA={self.SOURCE_RA:.4f}°, DEC={self.SOURCE_DEC:.4f}°")
                        print(f"  Altitude: {altaz.alt.deg:.2f}°")
                        print(f"  Azimuth: {altaz.az.deg:.2f}°")
                        print(f"  Zenith Angle: {zenith_angle:.2f}°")

                except Exception as e:
                    print(
                        f"Warning: Zenith calculation failed for image {i}: {e}")
                    pass

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

        df = pd.DataFrame(results)

        # Derived calculations
        df['C. Center (X)'] = (df['Ax'] + df['Bx'] + df['Cx'] + df['Dx']) / 4
        df['C. Center (Y)'] = (df['Ay'] + df['By'] + df['Cy'] + df['Dy']) / 4

        # Scale factor and corrections
        df['scale_factor'] = 16 / (df['By'] - df['Ay'])
        df['Correction_X'] = (
            df['Img_X'] - df['C. Center (X)']) * df['scale_factor'] * 0.318 * 60
        df['Correction_Y'] = (
            df['C. Center (Y)'] - df['Img_Y']) * df['scale_factor'] * 0.318 * 60
        numeric_columns = ['Ax', 'Ay', 'Bx', 'By', 'Cx', 'Cy', 'Dx', 'Dy',
                           'Img_X', 'Img_Y', 'C. Center (X)', 'C. Center (Y)',
                           'Correction_X', 'Correction_Y', 'scale_factor']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(4)
        return df

    def poly_fit_chi2(self, x, y, degree=3):
        """Polynomial fit with chi-squared"""
        if len(x) < degree + 2:
            return None, None, np.nan

        coeffs, _ = curve_fit(
            lambda x, *p: np.polyval(p[::-1], x), x, y, p0=np.ones(degree + 1))
        y_pred = np.polyval(coeffs[::-1], x)
        residuals = y - y_pred
        chi2_stat = np.sum(residuals**2)
        dof = len(x) - (degree + 1)
        reduced_chi2 = chi2_stat / dof if dof > 0 else np.nan
        return lambda z: np.polyval(
            coeffs[::-1], z), coeffs[::-1], reduced_chi2

    def plot_fits(self, df, pre_orders=None, post_orders=None):
        """Pre/Post Transit SEPARATE ANALYSIS with flexible polynomial orders"""
    
        # Default orders
        if pre_orders is None:
            pre_orders = [3]  # Original was 3rd order
        if post_orders is None:
            post_orders = [3]
        valid = ~(df['Correction_X'].isna() |
                  df['Correction_Y'].isna() | df['zenith'].isna())
        df_valid = df.loc[valid].copy()

        # ✅ IMPROVED: Find minimum zenith (transit point)
        transit_idx = df_valid['zenith'].idxmin()

        # Classify based on position relative to transit
        df_valid['Transit_Phase'] = 'Unknown'
        df_valid.loc[:transit_idx, 'Transit_Phase'] = 'Pre-Transit'
        df_valid.loc[transit_idx + 1:, 'Transit_Phase'] = 'Post-Transit'

        #print(f"\n✓ Transit detected at index {transit_idx}")
        print(f"  Min Zenith: {df_valid.loc[transit_idx, 'zenith']:.2f}°")
        #print(f"  Pre-transit points: {len(df_valid[df_valid['Transit_Phase']=='Pre-Transit'])}")
        #print(f"  Post-transit points: {len(df_valid[df_valid['Transit_Phase']=='Post-Transit'])}")

        pre_data = df_valid[df_valid['Transit_Phase'] == 'Pre-Transit']
        post_data = df_valid[df_valid['Transit_Phase'] == 'Post-Transit']
        has_pre = len(pre_data) > 3
        has_post = len(post_data) > 3

        # Create 2x2 or 1x2 plots
        if has_pre and has_post:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            axes = [axes[0], axes[1]]

        plot_idx = 0

        # PRE-TRANSIT PLOTS (if data exists)
        if has_pre:
            x_pre, y_pre, z_pre = pre_data['Correction_X'], pre_data['Correction_Y'], pre_data['zenith']
            z_fit = np.linspace(z_pre.min(), z_pre.max(), 100)
    
            # Colors for different orders
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
            # X Plot
            axes[plot_idx].scatter(z_pre, x_pre, c='orange', alpha=0.7, s=40, label='Data', zorder=1)
            #axes[plot_idx].scatter(z_pre, x_pre, c='black', alpha=1.0, s=30, label='Data', zorder=3)  #Black

            # Plot multiple polynomial fits and collect info for display
            fit_info_lines = []
            for idx, order in enumerate(pre_orders):
                if len(x_pre) > order:
                    try:
                        # Fit polynomial
                        coeffs = np.polyfit(z_pre, x_pre, order)
                        fit_func = np.poly1d(coeffs)
            
                        # Calculate chi-squared
                        y_pred = fit_func(z_pre)
                        residuals = x_pre - y_pred
                        chi2_stat = np.sum(residuals**2)
                        dof = len(x_pre) - (order + 1)
                        reduced_chi2 = chi2_stat / dof if dof > 0 else 0.0
            
                        # Plot fit line
                        color = colors[idx % len(colors)]
                        axes[plot_idx].plot(z_fit, fit_func(z_fit), color=color, 
                                           linewidth=2.5, label=f'Order {order}', zorder=2)
            
                        # Collect coefficient info
                        coeff_str = f"Order {order}: χ²/dof={reduced_chi2:.3f}"
                        for i, c in enumerate(coeffs[::-1]):  # Reverse to show X0, X1, X2...
                            coeff_str += f"\n  X{i}={c:.4f}"
                        fit_info_lines.append(coeff_str)
                    except:
                        pass

            # Display fit info box
            if fit_info_lines:
                info_text = "\n\n".join(fit_info_lines)
                axes[plot_idx].text(0.02, 0.98, info_text, transform=axes[plot_idx].transAxes, 
                                   fontsize=9, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_xlabel('Zenith Angle (deg)', fontsize=12)
            axes[plot_idx].set_ylabel('Correction_X (arcmin)', fontsize=12)
            axes[plot_idx].invert_xaxis()
            axes[plot_idx].legend(loc='upper right', fontsize=10)
            axes[plot_idx].set_title(f'Pre-Transit Correction_X - {self.SOURCE_NAME}', 
                                    fontsize=14, fontweight='bold')
            plot_idx += 1
            
            # Y Plot
            axes[plot_idx].scatter(z_pre, y_pre, c='red', alpha=0.7, s=40, label='Data', zorder=1)
            #axes[plot_idx].scatter(z_pre, y_pre, c='black', alpha=1.0, s=30, label='Data', zorder=3) #Black 
            fit_info_lines = []
            for idx, order in enumerate(pre_orders):
                if len(y_pre) > order:
                    try:
                        coeffs = np.polyfit(z_pre, y_pre, order)
                        fit_func = np.poly1d(coeffs)
            
                        y_pred = fit_func(z_pre)
                        residuals = y_pre - y_pred
                        chi2_stat = np.sum(residuals**2)
                        dof = len(y_pre) - (order + 1)
                        reduced_chi2 = chi2_stat / dof if dof > 0 else 0.0
            
                        color = colors[idx % len(colors)]
                        axes[plot_idx].plot(z_fit, fit_func(z_fit), color=color, 
                                           linewidth=2.5, label=f'Order {order}', zorder=2)
                    
                        coeff_str = f"Order {order}: χ²/dof={reduced_chi2:.3f}"
                        for i, c in enumerate(coeffs[::-1]):
                            coeff_str += f"\n  Y{i}={c:.4f}"
                        fit_info_lines.append(coeff_str)
                    except:
                        pass

            if fit_info_lines:
                info_text = "\n\n".join(fit_info_lines)
                axes[plot_idx].text(0.02, 0.98, info_text, transform=axes[plot_idx].transAxes, 
                                   fontsize=9, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_xlabel('Zenith Angle (deg)', fontsize=12)
            axes[plot_idx].set_ylabel('Correction_Y (arcmin)', fontsize=12)
            axes[plot_idx].invert_xaxis()
            axes[plot_idx].legend(loc='upper right', fontsize=10)
            axes[plot_idx].set_title(f'Pre-Transit Correction_Y - {self.SOURCE_NAME}', 
                                    fontsize=14, fontweight='bold')
            plot_idx += 1

        # POST-TRANSIT PLOTS (if data exists)
        if has_post:
            x_post, y_post, z_post = post_data['Correction_X'], post_data['Correction_Y'], post_data['zenith']
            z_fit = np.linspace(z_post.min(), z_post.max(), 100)
    
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
            # X Plot
            axes[plot_idx].scatter(z_post, x_post, c='blue', alpha=0.7, s=40, label='Data', zorder=1)
            #axes[plot_idx].scatter(z_post, x_post, c='black', alpha=1.0, s=30, label='Data', zorder=3)#Black
            fit_info_lines = []
            for idx, order in enumerate(post_orders):
                if len(x_post) > order:
                    try:
                        coeffs = np.polyfit(z_post, x_post, order)
                        fit_func = np.poly1d(coeffs)
                
                        y_pred = fit_func(z_post)
                        residuals = x_post - y_pred
                        chi2_stat = np.sum(residuals**2)
                        dof = len(x_post) - (order + 1)
                        reduced_chi2 = chi2_stat / dof if dof > 0 else 0.0
                
                        color = colors[idx % len(colors)]
                        axes[plot_idx].plot(z_fit, fit_func(z_fit), color=color, 
                                           linewidth=2.5, label=f'Order {order}', zorder=2)
                
                        coeff_str = f"Order {order}: χ²/dof={reduced_chi2:.3f}"
                        for i, c in enumerate(coeffs[::-1]):
                            coeff_str += f"\n  X{i}={c:.4f}"
                        fit_info_lines.append(coeff_str)
                    except:
                        pass
    
            if fit_info_lines:
                info_text = "\n\n".join(fit_info_lines)
                axes[plot_idx].text(0.02, 0.98, info_text, transform=axes[plot_idx].transAxes, 
                                   fontsize=9, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_xlabel('Zenith Angle (deg)', fontsize=12)
            axes[plot_idx].set_ylabel('Correction_X (arcmin)', fontsize=12)
            axes[plot_idx].legend(loc='best', fontsize=10)
            axes[plot_idx].set_title(f'Post-Transit Correction_X - {self.SOURCE_NAME}', 
                                    fontsize=14, fontweight='bold')
            plot_idx += 1

            # Y Plot
            axes[plot_idx].scatter(z_post, y_post, c='green', alpha=0.7, s=40, label='Data', zorder=1)
            #axes[plot_idx].scatter(z_post, y_post, c='black', alpha=1.0, s=30, label='Data', zorder=3) #Black
            fit_info_lines = []
            for idx, order in enumerate(post_orders):
                if len(y_post) > order:
                    try:
                        coeffs = np.polyfit(z_post, y_post, order)
                        fit_func = np.poly1d(coeffs)
                
                        y_pred = fit_func(z_post)
                        residuals = y_post - y_pred
                        chi2_stat = np.sum(residuals**2)
                        dof = len(y_post) - (order + 1)
                        reduced_chi2 = chi2_stat / dof if dof > 0 else 0.0
                        
                        color = colors[idx % len(colors)]
                        axes[plot_idx].plot(z_fit, fit_func(z_fit), color=color, 
                                           linewidth=2.5, label=f'Order {order}', zorder=2)
                
                        coeff_str = f"Order {order}: χ²/dof={reduced_chi2:.3f}"
                        for i, c in enumerate(coeffs[::-1]):
                            coeff_str += f"\n  Y{i}={c:.4f}"
                        fit_info_lines.append(coeff_str)
                    except:
                        pass
    
            if fit_info_lines:
                info_text = "\n\n".join(fit_info_lines)
                axes[plot_idx].text(0.02, 0.98, info_text, transform=axes[plot_idx].transAxes, 
                                   fontsize=9, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_xlabel('Zenith Angle (deg)', fontsize=12)
            axes[plot_idx].set_ylabel('Correction_Y (arcmin)', fontsize=12)
            axes[plot_idx].legend(loc='best', fontsize=10)
            axes[plot_idx].set_title(f'Post-Transit Correction_Y - {self.SOURCE_NAME}', 
                                    fontsize=14, fontweight='bold')

        # Developers Credit (bottom right corner - very small font)
        #fig.text(0.98, 0.02, 'Developed by Mkothari and Muskan Maheshwari',
                 #fontsize=8, fontstyle='italic', color='gray',
                 #verticalalignment='bottom', horizontalalignment='right',
                 #bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'pointing_fits_pre_post.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()  # Close figure to free memory

        print(f"Pre/Post Transit plots saved: {plot_path}")
        # Save detailed fit results to text file
        txt_path = os.path.join(self.output_dir, 'fit_results_pre_post.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TACTIC POINTING CORRECTION - FIT RESULTS\n")
            f.write(f"Source: {self.SOURCE_NAME}\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            if has_pre:
                f.write("=" * 70 + "\n")
                f.write("PRE-TRANSIT FITS (Zenith decreasing: 45° → 0°)\n")
                f.write("=" * 70 + "\n")
                f.write(f"Number of data points: {len(pre_data)}\n")
                f.write(
                    f"Zenith angle range: {z_pre.min():.2f}° to {z_pre.max():.2f}°\n\n")

                # Get fit parameters from pre-transit
                x_pre, y_pre, z_pre = pre_data['Correction_X'], pre_data['Correction_Y'], pre_data['zenith']
                fit_x, coeffs_x, chi2_x = self.poly_fit_chi2(z_pre, x_pre)
                if coeffs_x is None:
                    coeffs_x = [0, 0, 0, 0]
                    chi2_x = 0.0
                fit_y, coeffs_y, chi2_y = self.poly_fit_chi2(z_pre, y_pre)
                if coeffs_y is None:
                    coeffs_y = [0, 0, 0, 0]
                    chi2_y = 0.0
                f.write("Correction_X vs Zenith Angle:\n")
                f.write(
                    f"  Polynomial Fit: Correction_X = X0 + X1*z + X2*z² + X3*z³\n")
                f.write(f"  X0 = {coeffs_x[0]:.6f}\n")
                f.write(f"  X1 = {coeffs_x[1]:.6f}\n")
                f.write(f"  X2 = {coeffs_x[2]:.6f}\n")
                f.write(f"  X3 = {coeffs_x[3]:.6f}\n")
                f.write(f"  χ²/dof = {chi2_x:.4f}\n\n")

                f.write("Correction_Y vs Zenith Angle:\n")
                f.write(
                    f"  Polynomial Fit: Correction_Y = Y0 + Y1*z + Y2*z² + Y3*z³\n")
                f.write(f"  Y0 = {coeffs_y[0]:.6f}\n")
                f.write(f"  Y1 = {coeffs_y[1]:.6f}\n")
                f.write(f"  Y2 = {coeffs_y[2]:.6f}\n")
                f.write(f"  Y3 = {coeffs_y[3]:.6f}\n")
                f.write(f"  χ²/dof = {chi2_y:.4f}\n\n")

            if has_post:
                f.write("=" * 70 + "\n")
                f.write("POST-TRANSIT FITS (Zenith increasing: 0° → 45°)\n")
                f.write("=" * 70 + "\n")
                f.write(f"Number of data points: {len(post_data)}\n")
                f.write(
                    f"Zenith angle range: {z_post.min():.2f}° to {z_post.max():.2f}°\n\n")

                # Get fit parameters from post-transit
                x_post, y_post, z_post = post_data['Correction_X'], post_data['Correction_Y'], post_data['zenith']
                fit_x, coeffs_x, chi2_x = self.poly_fit_chi2(z_post, x_post)
                if coeffs_x is None:
                    coeffs_x = [0, 0, 0, 0]
                    chi2_x = 0.0
                fit_y, coeffs_y, chi2_y = self.poly_fit_chi2(z_post, y_post)
                if coeffs_y is None:
                    coeffs_y = [0, 0, 0, 0]
                    chi2_y = 0.0
                f.write("Correction_X vs Zenith Angle:\n")
                f.write(
                    f"  Polynomial Fit: Correction_X = X0 + X1*z + X2*z² + X3*z³\n")
                f.write(f"  X0 = {coeffs_x[0]:.6f}\n")
                f.write(f"  X1 = {coeffs_x[1]:.6f}\n")
                f.write(f"  X2 = {coeffs_x[2]:.6f}\n")
                f.write(f"  X3 = {coeffs_x[3]:.6f}\n")
                f.write(f"  χ²/dof = {chi2_x:.4f}\n\n")

                f.write("Correction_Y vs Zenith Angle:\n")
                f.write(
                    f"  Polynomial Fit: Correction_Y = Y0 + Y1*z + Y2*z² + Y3*z³\n")
                f.write(f"  Y0 = {coeffs_y[0]:.6f}\n")
                f.write(f"  Y1 = {coeffs_y[1]:.6f}\n")
                f.write(f"  Y2 = {coeffs_y[2]:.6f}\n")
                f.write(f"  Y3 = {coeffs_y[3]:.6f}\n")
                f.write(f"  χ²/dof = {chi2_y:.4f}\n\n")

            f.write("=" * 70 + "\n")
            f.write("END OF FIT RESULTS\n")
            f.write("=" * 70 + "\n")

        print(f"Fit results saved: {txt_path}")
    
    def _plot_fits_for_axis(self, ax, data, x_col, y_col, poly_orders, colors, title):
        """Helper to plot multiple polynomial fits on one axis"""
    
        valid_mask = data[y_col].notna()
        x_data = np.arange(len(data))[valid_mask]
        y_data = data[y_col].values[valid_mask]
    
        if len(x_data) < 2:
            return
    
        ax.scatter(x_data, y_data, color='white', s=50, 
                   alpha=0.6, label='Data', zorder=3)
    
        for idx, order in enumerate(poly_orders):
            if len(x_data) <= order:
                continue
        
            try:
                coeffs = np.polyfit(x_data, y_data, order)
                poly_fit = np.poly1d(coeffs)
            
                x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                y_smooth = poly_fit(x_smooth)
            
                color = colors[idx % len(colors)]
                ax.plot(x_smooth, y_smooth, 
                       color=color, linewidth=2,
                       label=f'Order {order}', zorder=2)
            except:
                continue
    
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Image Number', color='white', fontsize=12)
        ax.set_ylabel('Correction (arcsec)', color='white', fontsize=12)
        ax.legend(facecolor='#2d2d44', edgecolor='white', 
                 labelcolor='white', fontsize=10)
    def _save_fit_results_txt(self, df, pre_df, post_df, pre_orders, post_orders, ra_col='deltaRA', dec_col='deltaDEC'):
        """Save polynomial fit results to text file"""
        txt_path = os.path.join(self.output_dir, 'fit_results_pre_post.txt')
    
        with open(txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TACTIC Pointing Correction - Polynomial Fit Results\n")
            f.write("=" * 70 + "\n\n")
        
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images: {len(df)}\n")
            f.write(f"Transit Index: {df['zenith'].idxmin()}\n\n")
        
            # PRE-TRANSIT FITS
            f.write("=" * 70 + "\n")
            f.write("PRE-TRANSIT POLYNOMIAL FITS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Images: {len(pre_df)}\n")
            f.write(f"Fitting Orders: {pre_orders}\n\n")
        
            if len(pre_df) > max(pre_orders):
                for coord in [ra_col, dec_col]:
                    f.write(f"\n{coord} Fits:\n")
                    f.write("-" * 70 + "\n")
                
                    valid_mask = pre_df[coord].notna()
                    x_data = np.arange(len(pre_df))[valid_mask]
                    y_data = pre_df[coord].values[valid_mask]
                
                    for order in pre_orders:
                        if len(x_data) > order:
                            try:
                                coeffs = np.polyfit(x_data, y_data, order)
                                f.write(f"\nOrder {order}:\n")
                                f.write(f"  Coefficients: {coeffs}\n")
                            
                                # Calculate residuals
                                poly_fit = np.poly1d(coeffs)
                                residuals = y_data - poly_fit(x_data)
                                rms = np.sqrt(np.mean(residuals**2))
                                f.write(f"  RMS Residual: {rms:.4f} arcsec\n")
                            except:
                                f.write(f"\nOrder {order}: Fit failed\n")
        
            # POST-TRANSIT FITS
            f.write("\n" + "=" * 70 + "\n")
            f.write("POST-TRANSIT POLYNOMIAL FITS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Images: {len(post_df)}\n")
            f.write(f"Fitting Orders: {post_orders}\n\n")
        
            if len(post_df) > max(post_orders):
                for coord in [ra_col, dec_col]:
                    f.write(f"\n{coord} Fits:\n")
                    f.write("-" * 70 + "\n")
                
                    valid_mask = post_df[coord].notna()
                    x_data = np.arange(len(post_df))[valid_mask]
                    y_data = post_df[coord].values[valid_mask]
                
                    for order in post_orders:
                        if len(x_data) > order:
                            try:
                                coeffs = np.polyfit(x_data, y_data, order)
                                f.write(f"\nOrder {order}:\n")
                                f.write(f"  Coefficients: {coeffs}\n")
                            
                                poly_fit = np.poly1d(coeffs)
                                residuals = y_data - poly_fit(x_data)
                                rms = np.sqrt(np.mean(residuals**2))
                                f.write(f"  RMS Residual: {rms:.4f} arcsec\n")
                            except:
                                f.write(f"\nOrder {order}: Fit failed\n")
        
            f.write("\n" + "=" * 70 + "\n")
            f.write("End of Report\n")
            f.write("=" * 70 + "\n")
    def run_analysis(self):
        print("Parsing timestamps...")
        timestamps = self.parse_timestamps()
        fits_files = sorted([f for f in os.listdir(
            self.fits_dir) if f.endswith('.fit')])
        save_all = input(
            "Save ALL images for verification? (y/n): ").lower() == 'y'

        save_count = 5 if not save_all else len(fits_files)

        print(f"Saving {save_count} verification images...")
        print(f"Found {len(timestamps)} timestamps")
        analyzer.save_count = save_count  # Pass to class
        df = self.process_fits_files(timestamps)

        # Round coordinates to 4 decimal places
        coord_cols = ['Ax', 'Ay', 'Bx', 'By', 'Cx', 'Cy', 'Dx', 'Dy', 'Img_X', 'Img_Y',
                      'C. Center (X)', 'C. Center (Y)',
                      'Correction_X', 'Correction_Y']
        for col in coord_cols:
            if col in df.columns:
                df[col] = df[col].round(4)
        # Perfect Excel formatting
        df['zenith'] = df['zenith'].round(2)
        df['Correction_X'] = df['Correction_X'].round(4)
        df['Correction_Y'] = df['Correction_Y'].round(4)

        # Remove scale_factor column
        if 'scale_factor' in df.columns:
            df = df.drop('scale_factor', axis=1)

        # Reorder columns for clarity
        col_order = ['Time', 'zenith', 'Ax', 'Ay', 'Bx', 'By', 'Cx', 'Cy', 'Dx', 'Dy',
                     'Img_X', 'Img_Y', 'C. Center (X)', 'C. Center (Y)', 'Correction_X', 'Correction_Y']
        df = df[col_order]
        # Save in output_dir
        excel_path = os.path.join(
            self.output_dir,
            "pointing_correction_results.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\n✅ Excel saved: {excel_path}")

        detection_dir = os.path.join(self.output_dir, "detection_images")
        # df.to_excel("pointing_correction_results.xlsx", index=False)
        print(
            f"\nVerification images saved in 'detection_images/' folder ({save_count} files)")
        print("Check White/Colored circles match LED/Star centers!")
        print(f"\nExcel saved: {self.output_xls}")
        print("\nFirst 5 detection images saved as detection_000_*.png")
        print("Check these to verify LED/Star positions!")
        print("\nFirst 5 rows:")
        print(df[['Time', 'zenith', 'Ax', 'Ay', 'Img_X',
              'Img_Y', 'Correction_X', 'Correction_Y']].head())

        print("\nCreating plots...")
        self.plot_fits(df)
        print("Done! Check pointing_fits.png")
        print("Fit results saved to 'fit_results.txt'")
        return df


if __name__ == "__main__":
    analyzer = TACTICPointingCorrection()
    results = analyzer.run_analysis()
