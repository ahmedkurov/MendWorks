/*
  # Create indexes and utility functions

  1. Indexes
    - Add indexes for better query performance
    - Index on foreign keys and frequently queried columns

  2. Functions
    - Add function to calculate device health scores
    - Add function to update device status based on maintenance dates
*/

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_hospital_id ON users(hospital_id);
CREATE INDEX IF NOT EXISTS idx_devices_hospital_id ON devices(hospital_id);
CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);
CREATE INDEX IF NOT EXISTS idx_devices_next_maintenance ON devices(next_maintenance_date);
CREATE INDEX IF NOT EXISTS idx_maintenance_logs_device_id ON maintenance_logs(device_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_logs_date ON maintenance_logs(date_of_maintenance);

-- Function to calculate device risk score based on maintenance history
CREATE OR REPLACE FUNCTION calculate_device_risk_score(device_uuid uuid)
RETURNS numeric AS $$
DECLARE
    last_maintenance_date timestamptz;
    next_maintenance_date timestamptz;
    days_since_last numeric;
    days_until_next numeric;
    risk_score numeric := 0;
BEGIN
    -- Get device maintenance dates
    SELECT last_maintenance_date, next_maintenance_date
    INTO last_maintenance_date, next_maintenance_date
    FROM devices
    WHERE id = device_uuid;

    IF last_maintenance_date IS NULL OR next_maintenance_date IS NULL THEN
        RETURN 0.5; -- Default medium risk
    END IF;

    -- Calculate days since last maintenance
    days_since_last := EXTRACT(EPOCH FROM (now() - last_maintenance_date)) / 86400;
    -- Calculate days until next maintenance
    days_until_next := EXTRACT(EPOCH FROM (next_maintenance_date - now())) / 86400;

    -- Risk increases as we approach or pass the next maintenance date
    IF days_until_next <= 0 THEN
        risk_score := 1.0; -- High risk - overdue
    ELSIF days_until_next <= 7 THEN
        risk_score := 0.8; -- High risk - due soon
    ELSIF days_until_next <= 14 THEN
        risk_score := 0.6; -- Medium-high risk
    ELSIF days_until_next <= 30 THEN
        risk_score := 0.4; -- Medium risk
    ELSE
        risk_score := 0.2; -- Low risk
    END IF;

    -- Factor in time since last maintenance (older = higher risk)
    IF days_since_last > 90 THEN
        risk_score := LEAST(1.0, risk_score + 0.3);
    ELSIF days_since_last > 60 THEN
        risk_score := LEAST(1.0, risk_score + 0.2);
    ELSIF days_since_last > 30 THEN
        risk_score := LEAST(1.0, risk_score + 0.1);
    END IF;

    RETURN GREATEST(0, LEAST(1.0, risk_score));
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to update device status based on maintenance dates
CREATE OR REPLACE FUNCTION update_device_status()
RETURNS trigger AS $$
DECLARE
    days_until_maintenance numeric;
    new_status text;
BEGIN
    -- Calculate days until next maintenance
    days_until_maintenance := EXTRACT(EPOCH FROM (NEW.next_maintenance_date - now())) / 86400;

    -- Determine status based on days until maintenance
    IF days_until_maintenance <= 0 THEN
        new_status := 'Danger'; -- Overdue
    ELSIF days_until_maintenance <= 7 THEN
        new_status := 'Warning'; -- Due soon
    ELSE
        new_status := 'OK'; -- Good condition
    END IF;

    -- Update the status
    NEW.status := new_status;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update device status
DROP TRIGGER IF EXISTS trigger_update_device_status ON devices;
CREATE TRIGGER trigger_update_device_status
    BEFORE INSERT OR UPDATE OF next_maintenance_date
    ON devices
    FOR EACH ROW
    EXECUTE FUNCTION update_device_status();