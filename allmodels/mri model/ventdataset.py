import pandas as pd
import numpy as np
import random

# Reproducibility
np.random.seed(42)
random.seed(42)

def generate_ventilator_dataset(n_samples=2500):
    """
    Generate a synthetic ventilator dataset with correlated sensor fields.
    Fields:
      Airway_Pressure_cmH2O, Tidal_Volume_ml, Respiratory_Rate_bpm,
      Oxygen_Concentration_pct, Inspiratory_Flow_lpm, Exhaled_CO2_mmHg,
      Humidifier_Temperature_C, Compressor_Pressure_bar, Valve_Response_ms,
      Battery_Level_pct, Power_Consumption_kW, Alarm_Frequency_count,
      System_Runtime_hrs, Vibration_mm_s, Internal_Temperature_C,
      Filter_Status_pct, Failure (0/1), Cause_of_Failure, Condition,
      Time_to_Failure_days, Recommended_Solution
    """

    data = []
    conditions = ['Normal', 'Average', 'Warning', 'Critical']
    condition_weights = [0.40, 0.30, 0.20, 0.10]  # more normal cases
    
    for i in range(n_samples):
        condition = np.random.choice(conditions, p=condition_weights)

        # Base variables (we will make other variables depend on these)
        if condition == 'Normal':
            tidal_volume = np.random.uniform(400, 550)  # adult tidal volumes ~400-500ml
            resp_rate = np.random.uniform(12, 18)       # normal adult RR
            insp_flow = np.random.uniform(25, 55)       # common inspiratory flows
            fio2 = np.random.uniform(21, 40)            # low supplemental FiO2
            etco2 = np.random.normal(40, 2)             # EtCO2 ~35-45 mmHg
            humid_temp = np.random.uniform(34, 37)      # heated humidifier temp
            comp_pressure = np.random.uniform(3.5, 4.5) # medical gas supply ~3-4 bar
            valve_response = np.random.uniform(10, 50)  # ms (good valves are fast)
            battery = np.random.uniform(80, 100)        # backup battery healthy
            runtime = np.random.uniform(0, 3000)        # hours since last service
            filter_status = np.random.uniform(85, 100)
            vibration = 0.1 + runtime / 20000 + np.random.normal(0, 0.03)

            # Power consumption correlates with flow & humidifier heater use
            power_kw = 0.06 + 0.0012 * insp_flow + 0.002 * (humid_temp - 34) + np.random.normal(0, 0.01)
            alarm_lambda = 0.3

            failure = 0
            cause = 'None'
            ttf = np.random.uniform(200, 365)
            solution = 'Routine maintenance'

        elif condition == 'Average':
            tidal_volume = np.random.uniform(350, 450)
            resp_rate = np.random.uniform(14, 22)
            insp_flow = np.random.uniform(30, 70)
            fio2 = np.random.uniform(28, 60)
            etco2 = np.random.normal(42, 4)
            humid_temp = np.random.uniform(34, 38)
            comp_pressure = np.random.uniform(3.0, 4.2)
            valve_response = np.random.uniform(30, 100)
            battery = np.random.uniform(50, 90)
            runtime = np.random.uniform(2000, 7000)
            filter_status = np.random.uniform(65, 88)
            vibration = 0.3 + runtime / 15000 + np.random.normal(0, 0.08)

            power_kw = 0.08 + 0.0015 * insp_flow + 0.0025 * (humid_temp - 34) + np.random.normal(0, 0.02)
            alarm_lambda = 1.5

            failure = 0
            cause = 'None'
            ttf = np.random.uniform(100, 200)
            solution = 'Check filters, inspect valves, verify gas pressures'

        elif condition == 'Warning':
            tidal_volume = np.random.choice([
                np.random.uniform(250, 380),  # low-volume scenario
                np.random.uniform(500, 700)   # over-delivery scenario
            ])
            resp_rate = np.random.uniform(20, 30)
            insp_flow = np.random.uniform(40, 90)
            fio2 = np.random.uniform(40, 85)
            etco2 = np.random.normal(50, 7)  # drifted EtCO2
            humid_temp = np.random.uniform(36, 41)
            comp_pressure = np.random.uniform(2.5, 3.5)  # dropping gas pressure
            valve_response = np.random.uniform(60, 220)  # slower valves / latency
            battery = np.random.uniform(20, 70)
            runtime = np.random.uniform(6000, 10000)
            filter_status = np.random.uniform(30, 75)
            vibration = 0.8 + runtime / 9000 + np.random.normal(0, 0.25)

            power_kw = 0.12 + 0.002 * insp_flow + 0.003 * (humid_temp - 34) + np.random.normal(0, 0.03)
            alarm_lambda = 6

            # occasional failure in warning state (rare)
            failure = 0 if np.random.rand() > 0.06 else 1
            cause = 'None' if failure == 0 else 'Sensor Fault'
            ttf = np.random.uniform(10, 90) if failure else np.random.uniform(30, 100)
            solution = 'Inspect sensors, check gas supply & filters' if failure else 'Increased monitoring; service soon'

        else:  # Critical
            # produce extreme or anomalous values (either too low or too high)
            if np.random.rand() < 0.5:
                tidal_volume = np.random.uniform(100, 300)   # severe low-volume / leak
            else:
                tidal_volume = np.random.uniform(600, 900)   # severe over-delivery

            resp_rate = np.random.uniform(30, 60)
            insp_flow = np.random.uniform(10, 120)
            fio2 = np.random.uniform(70, 100)
            etco2 = np.random.uniform(20, 80)
            # humidifier may be failing (too low/high)
            if np.random.rand() < 0.5:
                humid_temp = np.random.uniform(20, 33)  # too cold (heater off/failure)
            else:
                humid_temp = np.random.uniform(42, 60)  # overheating
            comp_pressure = np.random.uniform(0.5, 3.0)  # low gas pressure in many critical cases
            valve_response = np.random.uniform(120, 800)  # severe latency / stuck valves
            battery = np.random.uniform(0, 30)  # near-dead battery
            runtime = np.random.uniform(9000, 20000)
            filter_status = np.random.uniform(0, 40)
            vibration = 2.5 + runtime / 6000 + np.random.normal(0, 0.6)

            power_kw = max(0.01, 0.03 + 0.003 * insp_flow + 0.005 * (humid_temp - 34) + np.random.normal(0, 0.05))
            alarm_lambda = 18

            failure = 1

            # Determine likely cause(s) of failure from sensors (priority order)
            if battery < 8:
                cause = 'Battery Failure / Power Loss'
            elif comp_pressure < 1.5:
                cause = 'Gas Supply Failure (Low Pressure)'
            elif valve_response > 300:
                cause = 'Valve Malfunction'
            elif filter_status < 15:
                cause = 'Filter Blockage'
            elif humid_temp > 50 or (humid_temp < 28):
                cause = 'Humidifier / Heater Failure (Over/Under Temp)'
            elif vibration > 6.0:
                cause = 'Mechanical Wear'
            elif etco2 > 70 or etco2 < 25:
                cause = 'Circuit / Patient Interface Fault'
            else:
                cause = 'Software / Unknown'

            ttf = np.random.uniform(0.1, 20)

            # recommended solution map for critical causes
            if cause == 'Battery Failure / Power Loss':
                solution = 'Switch to mains / replace battery immediately'
            elif cause == 'Gas Supply Failure (Low Pressure)':
                solution = 'Check pipeline & cylinder pressures; switch supply'
            elif cause == 'Valve Malfunction':
                solution = 'Replace/repair inspiratory/expiratory valve assembly'
            elif cause == 'Filter Blockage':
                solution = 'Replace intake / bacterial filter'
            elif cause == 'Humidifier / Heater Failure (Over/Under Temp)':
                solution = 'Service humidifier heater, check sensor'
            elif cause == 'Mechanical Wear':
                solution = 'Schedule urgent mechanical maintenance'
            elif cause == 'Circuit / Patient Interface Fault':
                solution = 'Inspect tubing, ETT, and connectors; verify patient circuit'
            else:
                solution = 'Contact vendor service; full diagnostics'

        # Correlated derived features
        # Airway pressure depends on tidal volume, inspiratory flow, and filter obstruction
        # formula tuned to produce pressures in typical clinical ranges (cmH2O)
        # base pressure + contributions:
        airway_pressure = (
            6.0 +
            0.02 * (tidal_volume - 500) +               # larger volumes -> higher pressure
            0.06 * (insp_flow - 40) +                    # higher flow -> slightly higher PIP
            (1.0 - filter_status / 100.0) * 8.0 +        # clogged filter increases pressure
            np.random.normal(0, 1.2)
        )

        # Alarm frequency: Poisson around lambda scaled by condition
        alarm_count = np.random.poisson(max(0.1, alarm_lambda))

        # Clamp values to realistic ranges
        tidal_volume = np.clip(tidal_volume, 50, 2000)               # ml
        resp_rate = np.clip(resp_rate, 1, 80)                        # bpm
        insp_flow = np.clip(insp_flow, 1, 200)                       # L/min
        fio2 = np.clip(fio2, 21, 100)                                # %
        etco2 = np.clip(etco2, 5, 150)                               # mmHg (extreme clipped)
        humid_temp = np.clip(humid_temp, 10, 80)                     # C
        comp_pressure = np.clip(comp_pressure, 0.1, 20)              # bar
        valve_response = np.clip(valve_response, 1, 2000)            # ms
        battery = np.clip(battery, 0, 100)                           # %
        power_kw = np.clip(power_kw, 0.005, 1.2)                     # kW (5W to 1200W)
        airway_pressure = np.clip(airway_pressure, 0.5, 120)         # cmH2O
        vibration = np.clip(vibration, 0.01, 12.0)                   # mm/s
        internal_temp = np.clip(28 + (power_kw * 30) + np.random.normal(0, 2), 20, 90)  # device internal temp
        filter_status = np.clip(filter_status, 0, 100)
        runtime = np.clip(runtime, 0, 200000)

        data.append([
            airway_pressure, tidal_volume, resp_rate, fio2, insp_flow,
            etco2, humid_temp, comp_pressure, valve_response, battery,
            power_kw, alarm_count, runtime, vibration, internal_temp,
            filter_status, failure, cause, condition, ttf, solution
        ])

    columns = [
        'Airway_Pressure_cmH2O', 'Tidal_Volume_ml', 'Respiratory_Rate_bpm',
        'Oxygen_Concentration_pct', 'Inspiratory_Flow_lpm', 'Exhaled_CO2_mmHg',
        'Humidifier_Temperature_C', 'Compressor_Pressure_bar', 'Valve_Response_ms',
        'Battery_Level_pct', 'Power_Consumption_kW', 'Alarm_Frequency_count',
        'System_Runtime_hrs', 'Vibration_mm_s', 'Internal_Temperature_C',
        'Filter_Status_pct', 'Failure', 'Cause_of_Failure', 'Condition',
        'Time_to_Failure_days', 'Recommended_Solution'
    ]

    df = pd.DataFrame(data, columns=columns)
    return df


if __name__ == "__main__":
    print("🔄 Generating synthetic ventilator dataset...")
    vent_df = generate_ventilator_dataset(n_samples=2500)

    print(f"📊 Dataset created with {len(vent_df)} samples")
    print("\n📈 Condition distribution:")
    print(vent_df['Condition'].value_counts())

    print("\n🚨 Failure distribution:")
    print(vent_df['Failure'].value_counts())

    # Save to CSV
    vent_df.to_csv('synthetic_ventilator_dataset.csv', index=False)
    print("\n💾 Dataset saved as 'synthetic_ventilator_dataset.csv'")
