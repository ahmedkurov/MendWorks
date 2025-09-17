import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://placeholder.supabase.co'
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBsYWNlaG9sZGVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NDUxOTI4MDAsImV4cCI6MTk2MDc2ODgwMH0.placeholder'

export const supabase = createClient(supabaseUrl, supabaseKey)

export type Database = {
  public: {
    Tables: {
      hospitals: {
        Row: {
          id: string
          name: string
          address: string
          contact_number: string
          created_at: string
        }
        Insert: {
          id?: string
          name: string
          address: string
          contact_number: string
          created_at?: string
        }
        Update: {
          id?: string
          name?: string
          address?: string
          contact_number?: string
          created_at?: string
        }
      }
      users: {
        Row: {
          id: string
          username: string
          hospital_id: string
          contact_info: string
          role: string
          created_at: string
        }
        Insert: {
          id: string
          username: string
          hospital_id: string
          contact_info: string
          role?: string
          created_at?: string
        }
        Update: {
          id?: string
          username?: string
          hospital_id?: string
          contact_info?: string
          role?: string
          created_at?: string
        }
      }
      devices: {
        Row: {
          id: string
          hospital_id: string
          device_type: 'MRI Scanner' | 'Ventilator' | 'EEG Machine'
          location: string
          status: 'OK' | 'Warning' | 'Danger'
          next_maintenance_date: string
          technician_id: string
          last_maintenance_date: string
          created_at: string
        }
        Insert: {
          id?: string
          hospital_id: string
          device_type: 'MRI Scanner' | 'Ventilator' | 'EEG Machine'
          location: string
          status?: 'OK' | 'Warning' | 'Danger'
          next_maintenance_date?: string
          technician_id: string
          last_maintenance_date?: string
          created_at?: string
        }
        Update: {
          id?: string
          hospital_id?: string
          device_type?: 'MRI Scanner' | 'CT Scanner' | 'Ventilator' | 'EEG Machine'
          location?: string
          status?: 'OK' | 'Warning' | 'Danger'
          next_maintenance_date?: string
          technician_id?: string
          last_maintenance_date?: string
          created_at?: string
        }
      }
      maintenance_logs: {
        Row: {
          id: string
          device_id: string
          uploaded_by: string
          date_of_maintenance: string
          log_file_url: string | null
          notes: string | null
          created_at: string
        }
        Insert: {
          id?: string
          device_id: string
          uploaded_by: string
          date_of_maintenance: string
          log_file_url?: string | null
          notes?: string | null
          created_at?: string
        }
        Update: {
          id?: string
          device_id?: string
          uploaded_by?: string
          date_of_maintenance?: string
          log_file_url?: string | null
          notes?: string | null
          created_at?: string
        }
      }
    }
  }
}