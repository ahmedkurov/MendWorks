import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_mri_dataset(n_samples=2500):
    """Generate synthetic MRI machine dataset with realistic correlations"""
    data = []
    
    # Generate samples for each condition
    conditions = ['Normal', 'Average', 'Warning', 'Critical']
    condition_weights = [0.35, 0.30, 0.20, 0.15]  # Distribution of conditions
    
    for i in range(n_samples):
        # Select condition based on weights
        condition = np.random.choice(conditions, p=condition_weights)
        
        # Generate correlated sensor values
        if condition == 'Normal':
            # Normal operation - all sensors in normal range
            helium_level = np.random.uniform(87, 92)
            magnetic_field = 3.0 + (helium_level - 89.5) * 0.002 + np.random.normal(0, 0.005)
            
            rf_power = np.random.uniform(27, 32)
            coil_temp = 25 + (rf_power - 29.5) * 0.8 + np.random.normal(0, 1.5)
            power_consumption = 53 + (rf_power - 29.5) * 2.2 + np.random.normal(0, 2)
            
            system_runtime = np.random.uniform(0, 5000)
            vibration = 0.85 + (system_runtime / 10000) + np.random.normal(0, 0.2)
            
            gradient_temp = np.random.uniform(20, 25)
            compressor_pressure = np.random.uniform(14, 17)
            chiller_flow = np.random.uniform(50, 60)
            humidity = np.random.uniform(45, 55)
            quench_risk = np.random.uniform(0.01, 0.05)
            
            failure = 0
            cause = 'None'
            ttf = np.random.uniform(200, 365)
            
        elif condition == 'Average':
            # Slight degradation
            helium_level = np.random.uniform(82, 87)
            magnetic_field = 3.0 + (helium_level - 84.5) * 0.003 + np.random.normal(0, 0.008)
            
            rf_power = np.random.uniform(32, 37)
            coil_temp = 25 + (rf_power - 34.5) * 0.9 + np.random.normal(0, 2)
            power_consumption = 53 + (rf_power - 34.5) * 2.5 + np.random.normal(0, 3)
            
            system_runtime = np.random.uniform(3000, 7000)
            vibration = 1.5 + (system_runtime / 8000) + np.random.normal(0, 0.3)
            
            gradient_temp = np.random.uniform(25, 30)
            compressor_pressure = np.random.uniform(12, 15)
            chiller_flow = np.random.uniform(45, 55)
            humidity = np.random.uniform(55, 65)
            quench_risk = np.random.uniform(0.05, 0.12)
            
            failure = 0
            cause = 'None'
            ttf = np.random.uniform(100, 200)
            
        elif condition == 'Warning':
            # Warning levels
            helium_level = np.random.uniform(73, 81)
            magnetic_field = 2.94 + (helium_level - 77) * 0.002 + np.random.normal(0, 0.01)
            
            rf_power = np.random.uniform(37, 43)
            coil_temp = 30 + (rf_power - 40) * 1.1 + np.random.normal(0, 2.5)
            power_consumption = 60 + (rf_power - 40) * 3 + np.random.normal(0, 4)
            
            system_runtime = np.random.uniform(6000, 9000)
            vibration = 2.7 + (system_runtime / 6000) + np.random.normal(0, 0.4)
            
            gradient_temp = np.random.uniform(30, 35)
            compressor_pressure = np.random.uniform(10, 13)
            chiller_flow = np.random.uniform(35, 45)
            humidity = np.random.uniform(60, 70)
            quench_risk = np.random.uniform(0.15, 0.25)
            
            failure = 0
            cause = 'None'
            ttf = np.random.uniform(50, 100)
            
        else:  # Critical
            # Critical failure imminent
            helium_level = np.random.uniform(60, 72)
            magnetic_field = 2.85 + (helium_level - 66) * 0.001 + np.random.normal(0, 0.015)
            
            rf_power = np.random.uniform(45, 55)
            coil_temp = 38 + (rf_power - 50) * 1.2 + np.random.normal(0, 3)
            power_consumption = 70 + (rf_power - 50) * 3.5 + np.random.normal(0, 5)
            
            system_runtime = np.random.uniform(8000, 12000)
            vibration = 4.5 + (system_runtime / 5000) + np.random.normal(0, 0.5)
            
            gradient_temp = np.random.uniform(40, 50)
            compressor_pressure = np.random.uniform(5, 9)
            chiller_flow = np.random.uniform(20, 30)
            humidity = np.random.uniform(75, 85)
            quench_risk = np.random.uniform(0.35, 0.50)
            
            failure = 1
            
            # Determine failure cause based on sensor values
            if helium_level < 65 or compressor_pressure < 8:
                cause = np.random.choice(['Cryogenic System Failure'])
            elif magnetic_field < 2.90 or quench_risk > 0.40:
                cause = np.random.choice(['Magnet Field Instability'])
            elif rf_power > 50 or coil_temp > 45:
                cause = np.random.choice(['RF System Overheating'])
            elif gradient_temp > 45:
                cause = np.random.choice(['Gradient System Failure'])
            elif power_consumption > 80:
                cause = np.random.choice(['Power System Failure'])
            elif vibration > 5.0 or system_runtime > 10000:
                cause = np.random.choice(['Mechanical Wear'])
            else:
                cause = np.random.choice(['Environmental Issues'])
                
            ttf = np.random.uniform(1, 30)
        
        # Clamp values to realistic ranges
        helium_level = np.clip(helium_level, 55, 95)
        magnetic_field = np.clip(magnetic_field, 2.5, 3.1)
        rf_power = np.clip(rf_power, 20, 60)
        gradient_temp = np.clip(gradient_temp, 15, 55)
        compressor_pressure = np.clip(compressor_pressure, 3, 20)
        chiller_flow = np.clip(chiller_flow, 15, 70)
        room_humidity = np.clip(humidity, 25, 90)
        vibration = np.clip(vibration, 0.1, 8.0)
        coil_temp = np.clip(coil_temp, 18, 55)
        power_consumption = np.clip(power_consumption, 40, 95)
        quench_risk = np.clip(quench_risk, 0.01, 0.6)
        
        # Determine recommended solution
        if failure == 1:
            if cause == 'Cryogenic System Failure':
                solution = 'Check helium supply and compressor'
            elif cause == 'Magnet Field Instability':
                solution = 'Inspect superconducting magnet'
            elif cause == 'RF System Overheating':
                solution = 'Service RF amplifiers and cooling'
            elif cause == 'Gradient System Failure':
                solution = 'Replace gradient coils'
            elif cause == 'Power System Failure':
                solution = 'Inspect power supply unit'
            elif cause == 'Mechanical Wear':
                solution = 'Schedule mechanical maintenance'
            else:
                solution = 'Control environmental conditions'
        else:
            solution = 'Routine maintenance'
        
        data.append([
            helium_level, magnetic_field, rf_power, gradient_temp,
            compressor_pressure, chiller_flow, room_humidity, system_runtime,
            vibration, coil_temp, power_consumption, quench_risk,
            failure, cause, condition, ttf, solution
        ])
    
    # Create DataFrame
    columns = [
        'Helium_Level_pct', 'Magnetic_Field_T', 'RF_Power_kW', 'Gradient_Temp_C',
        'Compressor_Pressure_bar', 'Chiller_Flow_lpm', 'Room_Humidity_pct', 'System_Runtime_hrs',
        'Vibration_mm_s', 'Coil_Temperature_C', 'Power_Consumption_kW', 'Magnet_Quench_Risk',
        'Failure', 'Cause_of_Failure', 'Condition', 'Time_to_Failure_days', 'Recommended_Solution'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate and save dataset
if __name__ == "__main__":
    print("🔄 Generating synthetic MRI machine dataset...")
    mri_df = generate_mri_dataset()
    
    print(f"📊 Dataset created with {len(mri_df)} samples")
    print(f"📈 Condition distribution:")
    print(mri_df['Condition'].value_counts())
    
    print(f"\n🚨 Failure distribution:")
    print(mri_df['Failure'].value_counts())
    
    # Save the dataset
    mri_df.to_csv('synthetic_mri_machine_dataset.csv', index=False)
    print("\n💾 Dataset saved as 'synthetic_mri_machine_dataset.csv'")
