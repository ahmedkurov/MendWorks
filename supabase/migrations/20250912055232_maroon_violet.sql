/*
  # Create maintenance logs table

  1. New Tables
    - `maintenance_logs`
      - `id` (uuid, primary key)
      - `device_id` (uuid, references devices)
      - `uploaded_by` (uuid, references users)
      - `date_of_maintenance` (timestamp, when maintenance was performed)
      - `log_file_url` (text, optional file URL)
      - `notes` (text, optional maintenance notes)
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on `maintenance_logs` table
    - Add policies for technicians to manage maintenance logs
*/

CREATE TABLE IF NOT EXISTS maintenance_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  device_id uuid REFERENCES devices(id) ON DELETE CASCADE,
  uploaded_by uuid REFERENCES users(id) ON DELETE SET NULL,
  date_of_maintenance timestamptz NOT NULL,
  log_file_url text,
  notes text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE maintenance_logs ENABLE ROW LEVEL SECURITY;

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