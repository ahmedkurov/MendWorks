import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import Layout from './components/Layout'
import ProtectedRoute from './components/ProtectedRoute'
import Login from './pages/Login'
import Register from './pages/Register'
import DeviceSetup from './pages/DeviceSetup'
import Dashboard from './pages/Dashboard'
import AddDevice from './pages/AddDevice'
import Settings from './pages/Settings'
import EEGMaintenanceLog from './pages/EEGMaintenanceLog'
import MRIMaintenanceLog from './pages/MRIMaintenanceLog'
import VentilatorMaintenanceLog from './pages/VentilatorMaintenanceLog'

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          
          {/* Protected routes */}
          <Route path="/device-setup" element={
            <ProtectedRoute>
              <DeviceSetup />
            </ProtectedRoute>
          } />
          
          <Route path="/" element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="add-device" element={<AddDevice />} />
            <Route path="settings" element={<Settings />} />
          </Route>

          {/* EEG Maintenance Log Route */}
          <Route path="/device/:deviceId/eeg-maintenance" element={
            <ProtectedRoute>
              <EEGMaintenanceLog />
            </ProtectedRoute>
          } />

          {/* MRI Maintenance Log Route */}
          <Route path="/device/:deviceId/mri-maintenance" element={
            <ProtectedRoute>
              <MRIMaintenanceLog />
            </ProtectedRoute>
          } />

          {/* Ventilator Maintenance Log Route */}
          <Route path="/device/:deviceId/ventilator-maintenance" element={
            <ProtectedRoute>
              <VentilatorMaintenanceLog />
            </ProtectedRoute>
          } />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  )
}

export default App