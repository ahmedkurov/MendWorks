/*
  # Create hospitals table

  1. New Tables
    - `hospitals`
      - `id` (uuid, primary key)
      - `name` (text, hospital name)
      - `address` (text, hospital address)
      - `contact_number` (text, hospital contact number)
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on `hospitals` table
    - Add policy for authenticated users to read/write their own hospital data
*/

CREATE TABLE IF NOT EXISTS hospitals (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  address text NOT NULL,
  contact_number text NOT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE hospitals ENABLE ROW LEVEL SECURITY;

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