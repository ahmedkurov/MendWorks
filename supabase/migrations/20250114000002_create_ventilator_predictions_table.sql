-- Create Ventilator Predictions Table
-- This table stores detailed AI prediction results for ventilators

-- 1. Create ventilator_predictions table
CREATE TABLE IF NOT EXISTS ventilator_predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  device_id uuid REFERENCES devices(id) ON DELETE CASCADE,
  maintenance_log_id uuid REFERENCES maintenance_logs(id) ON DELETE CASCADE,
  uploaded_by uuid REFERENCES users(id) ON DELETE SET NULL,
  
  -- Sensor readings used for prediction (matching frontend column names)
  airway_pressure numeric NOT NULL,
  tidal_volume numeric NOT NULL,
  respiratory_rate numeric NOT NULL,
  oxygen_concentration numeric NOT NULL,
  inspiratory_flow numeric NOT NULL,
  exhaled_co2 numeric NOT NULL,
  humidifier_temperature numeric NOT NULL,
  compressor_pressure numeric NOT NULL,
  valve_response numeric NOT NULL,
  battery_level numeric NOT NULL,
  power_consumption numeric NOT NULL,
  alarm_frequency numeric NOT NULL,
  system_runtime numeric NOT NULL,
  vibration numeric NOT NULL,
  internal_temperature numeric NOT NULL,
  filter_status numeric NOT NULL,
  
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

-- 2. Enable Row Level Security
ALTER TABLE ventilator_predictions ENABLE ROW LEVEL SECURITY;

-- 3. Create RLS Policies for ventilator_predictions
CREATE POLICY "Users can read Ventilator predictions for hospital devices"
  ON ventilator_predictions
  FOR SELECT
  TO authenticated
  USING (
    device_id IN (
      SELECT d.id FROM devices d
      INNER JOIN users u ON d.hospital_id = u.hospital_id
      WHERE u.id = auth.uid()
    )
  );

CREATE POLICY "Users can insert Ventilator predictions for hospital devices"
  ON ventilator_predictions
  FOR INSERT
  TO authenticated
  WITH CHECK (
    device_id IN (
      SELECT d.id FROM devices d
      INNER JOIN users u ON d.hospital_id = u.hospital_id
      WHERE u.id = auth.uid()
    )
  );

CREATE POLICY "Users can update own Ventilator predictions"
  ON ventilator_predictions
  FOR UPDATE
  TO authenticated
  USING (uploaded_by = auth.uid());

-- 4. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_ventilator_predictions_device_id ON ventilator_predictions(device_id);
CREATE INDEX IF NOT EXISTS idx_ventilator_predictions_maintenance_log_id ON ventilator_predictions(maintenance_log_id);
CREATE INDEX IF NOT EXISTS idx_ventilator_predictions_prediction_date ON ventilator_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_ventilator_predictions_condition ON ventilator_predictions(predicted_condition);
CREATE INDEX IF NOT EXISTS idx_ventilator_predictions_confidence ON ventilator_predictions(confidence_score);

-- 5. Create function to get latest Ventilator prediction for a device
CREATE OR REPLACE FUNCTION get_latest_ventilator_prediction(device_uuid uuid)
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
    vp.id,
    vp.predicted_condition,
    vp.predicted_failure,
    vp.predicted_cause,
    vp.recommended_solution,
    vp.time_to_failure_days,
    vp.confidence_score,
    vp.class_probabilities,
    vp.prediction_date
  FROM ventilator_predictions vp
  WHERE vp.device_id = device_uuid
  ORDER BY vp.prediction_date DESC
  LIMIT 1;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 6. Create function to update device status based on Ventilator prediction
CREATE OR REPLACE FUNCTION update_device_status_from_ventilator_prediction()
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

-- 7. Create trigger to automatically update device status when Ventilator prediction is inserted
DROP TRIGGER IF EXISTS trigger_update_device_from_ventilator_prediction ON ventilator_predictions;
CREATE TRIGGER trigger_update_device_from_ventilator_prediction
    AFTER INSERT ON ventilator_predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_device_status_from_ventilator_prediction();

-- Success message
SELECT 'Ventilator Predictions table created successfully! AI predictions will now be stored in structured format.' as message;