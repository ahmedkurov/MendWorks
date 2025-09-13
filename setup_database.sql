-- MendWorks Database Setup Script
-- Run this in your Supabase SQL Editor

-- 1. Create hospitals table
CREATE TABLE IF NOT EXISTS hospitals (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  address text NOT NULL,
  contact_number text NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- 2. Create users table
CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY REFERENCES auth.users ON DELETE CASCADE,
  username text NOT NULL,
  hospital_id uuid REFERENCES hospitals(id) ON DELETE CASCADE,
  contact_info text NOT NULL,
  role text DEFAULT 'technician',
  created_at timestamptz DEFAULT now()
);

-- 3. Create devices table
CREATE TABLE IF NOT EXISTS devices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  hospital_id uuid REFERENCES hospitals(id) ON DELETE CASCADE,
  device_type text NOT NULL CHECK (device_type IN ('MRI Scanner', 'CT Scanner', 'Ventilator')),
  location text NOT NULL,
  status text DEFAULT 'OK' CHECK (status IN ('OK', 'Warning', 'Danger')),
  next_maintenance_date timestamptz DEFAULT (now() + interval '30 days'),
  technician_id uuid REFERENCES users(id) ON DELETE SET NULL,
  last_maintenance_date timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

-- 4. Create maintenance_logs table
CREATE TABLE IF NOT EXISTS maintenance_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  device_id uuid REFERENCES devices(id) ON DELETE CASCADE,
  uploaded_by uuid REFERENCES users(id) ON DELETE SET NULL,
  date_of_maintenance timestamptz NOT NULL,
  log_file_url text,
  notes text,
  created_at timestamptz DEFAULT now()
);

-- 5. Enable Row Level Security
ALTER TABLE hospitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE maintenance_logs ENABLE ROW LEVEL SECURITY;

-- 6. Create RLS Policies for hospitals
CREATE POLICY "Users can read hospitals"
  ON hospitals
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Users can create hospitals"
  ON hospitals
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Users can update hospitals"
  ON hospitals
  FOR UPDATE
  TO authenticated
  USING (true);

-- 7. Create RLS Policies for users
CREATE POLICY "Users can read own profile"
  ON users
  FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON users
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON users
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = id);

-- 8. Create RLS Policies for devices
CREATE POLICY "Users can read hospital devices"
  ON devices
  FOR SELECT
  TO authenticated
  USING (
    hospital_id IN (
      SELECT hospital_id FROM users WHERE id = auth.uid()
    )
  );

CREATE POLICY "Users can insert hospital devices"
  ON devices
  FOR INSERT
  TO authenticated
  WITH CHECK (
    hospital_id IN (
      SELECT hospital_id FROM users WHERE id = auth.uid()
    )
  );

CREATE POLICY "Users can update hospital devices"
  ON devices
  FOR UPDATE
  TO authenticated
  USING (
    hospital_id IN (
      SELECT hospital_id FROM users WHERE id = auth.uid()
    )
  );

-- 9. Create RLS Policies for maintenance_logs
CREATE POLICY "Users can read maintenance logs for hospital devices"
  ON maintenance_logs
  FOR SELECT
  TO authenticated
  USING (
    device_id IN (
      SELECT d.id FROM devices d
      INNER JOIN users u ON d.hospital_id = u.hospital_id
      WHERE u.id = auth.uid()
    )
  );

CREATE POLICY "Users can insert maintenance logs for hospital devices"
  ON maintenance_logs
  FOR INSERT
  TO authenticated
  WITH CHECK (
    device_id IN (
      SELECT d.id FROM devices d
      INNER JOIN users u ON d.hospital_id = u.hospital_id
      WHERE u.id = auth.uid()
    )
  );

CREATE POLICY "Users can update own maintenance logs"
  ON maintenance_logs
  FOR UPDATE
  TO authenticated
  USING (uploaded_by = auth.uid());

-- 10. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_hospital_id ON users(hospital_id);
CREATE INDEX IF NOT EXISTS idx_devices_hospital_id ON devices(hospital_id);
CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);
CREATE INDEX IF NOT EXISTS idx_devices_next_maintenance ON devices(next_maintenance_date);
CREATE INDEX IF NOT EXISTS idx_maintenance_logs_device_id ON maintenance_logs(device_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_logs_date ON maintenance_logs(date_of_maintenance);

-- 11. Create function to calculate device risk score
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

-- 12. Create function to update device status
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

-- 13. Create trigger to automatically update device status
DROP TRIGGER IF EXISTS trigger_update_device_status ON devices;
CREATE TRIGGER trigger_update_device_status
    BEFORE INSERT OR UPDATE OF next_maintenance_date
    ON devices
    FOR EACH ROW
    EXECUTE FUNCTION update_device_status();

-- 14. Insert sample data for testing
INSERT INTO hospitals (id, name, address, contact_number) VALUES 
('550e8400-e29b-41d4-a716-446655440000', 'Demo Hospital', '123 Medical Center Dr, City, State 12345', '(555) 123-4567')
ON CONFLICT (id) DO NOTHING;

-- Success message
SELECT 'Database setup completed successfully! You can now use the MendWorks application.' as message;
