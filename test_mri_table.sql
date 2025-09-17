-- Test script to check if MRI predictions table exists and is accessible
-- Run this in your Supabase SQL Editor

-- 1. Check if the table exists
SELECT table_name, table_schema 
FROM information_schema.tables 
WHERE table_name = 'mri_predictions';

-- 2. Check table structure
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'mri_predictions'
ORDER BY ordinal_position;

-- 3. Check RLS policies
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual
FROM pg_policies 
WHERE tablename = 'mri_predictions';

-- 4. Test inserting a sample record (this will fail if table doesn't exist or RLS blocks it)
-- First, let's get a valid device_id and user_id
SELECT d.id as device_id, u.id as user_id, d.hospital_id
FROM devices d
JOIN users u ON d.hospital_id = u.hospital_id
WHERE d.device_type = 'MRI Scanner'
LIMIT 1;

-- 5. If we have a device, try to insert a test prediction
-- (This will be commented out initially - uncomment if you want to test insertion)
/*
INSERT INTO mri_predictions (
  device_id,
  maintenance_log_id,
  uploaded_by,
  helium_level,
  magnetic_field,
  rf_power,
  gradient_temp,
  compressor_pressure,
  chiller_flow,
  room_humidity,
  system_runtime,
  vibration,
  coil_temperature,
  power_consumption,
  magnet_quench_risk,
  predicted_condition,
  predicted_failure,
  predicted_cause,
  recommended_solution,
  time_to_failure_days,
  confidence_score,
  class_probabilities
) VALUES (
  'your-device-id-here',
  'your-maintenance-log-id-here',
  'your-user-id-here',
  89.5,
  3.0,
  30.0,
  22.0,
  15.0,
  55.0,
  50.0,
  2000.0,
  1.0,
  25.0,
  50.0,
  0.05,
  'Good',
  'No',
  'Normal Operation',
  'Continue monitoring',
  150.0,
  85.0,
  '{"Good": 85.0, "Average": 10.0, "Warning": 3.0, "Critical": 2.0}'::jsonb
);
*/