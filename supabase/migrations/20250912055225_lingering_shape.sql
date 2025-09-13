/*
  # Create devices table

  1. New Tables
    - `devices`
      - `id` (uuid, primary key)
      - `hospital_id` (uuid, references hospitals)
      - `device_type` (text, type of medical device)
      - `location` (text, device location in hospital)
      - `status` (text, current status: OK, Warning, Danger)
      - `next_maintenance_date` (timestamp, predicted maintenance date)
      - `technician_id` (uuid, references users)
      - `last_maintenance_date` (timestamp, last maintenance performed)
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on `devices` table
    - Add policies for hospital technicians to manage devices
*/

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

ALTER TABLE devices ENABLE ROW LEVEL SECURITY;

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