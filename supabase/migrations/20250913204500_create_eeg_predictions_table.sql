-- Create EEG Predictions Table
-- This table stores detailed AI prediction results for EEG machines

-- 1. Create eeg_predictions table
CREATE TABLE IF NOT EXISTS eeg_predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  device_id uuid REFERENCES devices(id) ON DELETE CASCADE,
  maintenance_log_id uuid REFERENCES maintenance_logs(id) ON DELETE CASCADE,
  uploaded_by uuid REFERENCES users(id) ON DELETE SET NULL,
  
  -- Sensor readings used for prediction
  temperature numeric NOT NULL,
  voltage numeric NOT NULL,
  current numeric NOT NULL,
  vibration numeric NOT NULL,
  pressure numeric NOT NULL,
  humidity numeric NOT NULL,
  usage_hours numeric NOT NULL,
  
  -- AI prediction results
  predicted_condition text NOT NULL CHECK (predicted_condition IN ('Good', 'Average', 'Warning', 'Critical')),
  predicted_failure text NOT NULL CHECK (predicted_failure IN ('Yes', 'No')),
  predicted_cause text NOT NULL,
  recommended_solution text NOT NULL,
  time_to_failure_days numeric NOT NULL,
  confidence_score numeric NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 100),
  
  -- Class probabilities (stored as JSON)
  class_probabilities jsonb NOT NULL,
  
  -- Timestamps
  prediction_date timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

-- 2. Update devices table to include EEG Machine
ALTER TABLE devices 
DROP CONSTRAINT IF EXISTS devices_device_type_check;

ALTER TABLE devices 
ADD CONSTRAINT devices_device_type_check 
CHECK (device_type IN ('MRI Scanner', 'CT Scanner', 'Ventilator', 'EEG Machine'));

-- 3. Enable Row Level Security
ALTER TABLE eeg_predictions ENABLE ROW LEVEL SECURITY;

-- 4. Create RLS Policies for eeg_predictions
CREATE POLICY "Users can read EEG predictions for hospital devices"
  ON eeg_predictions
  FOR SELECT
  TO authenticated
  USING (
    device_id IN (
      SELECT d.id FROM devices d
      INNER JOIN users u ON d.hospital_id = u.hospital_id
      WHERE u.id = auth.uid()
    )
  );

CREATE POLICY "Users can insert EEG predictions for hospital devices"
  ON eeg_predictions
  FOR INSERT
  TO authenticated
  WITH CHECK (
    device_id IN (
      SELECT d.id FROM devices d
      INNER JOIN users u ON d.hospital_id = u.hospital_id
      WHERE u.id = auth.uid()
    )
  );

CREATE POLICY "Users can update own EEG predictions"
  ON eeg_predictions
  FOR UPDATE
  TO authenticated
  USING (uploaded_by = auth.uid());

-- 5. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_eeg_predictions_device_id ON eeg_predictions(device_id);
CREATE INDEX IF NOT EXISTS idx_eeg_predictions_maintenance_log_id ON eeg_predictions(maintenance_log_id);
CREATE INDEX IF NOT EXISTS idx_eeg_predictions_prediction_date ON eeg_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_eeg_predictions_condition ON eeg_predictions(predicted_condition);
CREATE INDEX IF NOT EXISTS idx_eeg_predictions_confidence ON eeg_predictions(confidence_score);

-- 6. Create function to get latest EEG prediction for a device
CREATE OR REPLACE FUNCTION get_latest_eeg_prediction(device_uuid uuid)
RETURNS TABLE (
  id uuid,
  predicted_condition text,
  predicted_failure text,
  predicted_cause text,
  recommended_solution text,
  time_to_failure_days numeric,
  confidence_score numeric,
  class_probabilities jsonb,
  prediction_date timestamptz
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    ep.id,
    ep.predicted_condition,
    ep.predicted_failure,
    ep.predicted_cause,
    ep.recommended_solution,
    ep.time_to_failure_days,
    ep.confidence_score,
    ep.class_probabilities,
    ep.prediction_date
  FROM eeg_predictions ep
  WHERE ep.device_id = device_uuid
  ORDER BY ep.prediction_date DESC
  LIMIT 1;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 7. Create function to update device status based on EEG prediction
CREATE OR REPLACE FUNCTION update_device_status_from_eeg_prediction()
RETURNS trigger AS $$
DECLARE
    new_status text;
BEGIN
    -- Map AI prediction condition to device status
    CASE NEW.predicted_condition
        WHEN 'Good' THEN new_status := 'OK';
        WHEN 'Average' THEN new_status := 'OK';
        WHEN 'Warning' THEN new_status := 'Warning';
        WHEN 'Critical' THEN new_status := 'Danger';
        ELSE new_status := 'OK';
    END CASE;

    -- Update the device status
    UPDATE devices 
    SET 
        status = new_status,
        last_maintenance_date = NEW.prediction_date,
        next_maintenance_date = NEW.prediction_date + (NEW.time_to_failure_days || ' days')::interval
    WHERE id = NEW.device_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 8. Create trigger to automatically update device status when EEG prediction is inserted
DROP TRIGGER IF EXISTS trigger_update_device_from_eeg_prediction ON eeg_predictions;
CREATE TRIGGER trigger_update_device_from_eeg_prediction
    AFTER INSERT ON eeg_predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_device_status_from_eeg_prediction();

-- Success message
SELECT 'EEG Predictions table created successfully! AI predictions will now be stored in structured format.' as message;
