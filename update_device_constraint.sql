-- Update device_type constraint to include EEG Machine
-- Run this in your Supabase SQL Editor

-- First, drop the existing constraint
ALTER TABLE devices DROP CONSTRAINT IF EXISTS devices_device_type_check;

-- Add the new constraint that includes EEG Machine
ALTER TABLE devices ADD CONSTRAINT devices_device_type_check 
CHECK (device_type IN ('MRI Scanner', 'CT Scanner', 'Ventilator', 'EEG Machine'));

-- Verify the constraint was updated successfully
SELECT 'Device type constraint updated successfully! EEG Machine is now allowed.' as message;
