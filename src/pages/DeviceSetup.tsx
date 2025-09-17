import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { Activity, Monitor, Heart, Wind, MapPin, Plus, X, Brain } from 'lucide-react'

const deviceTypes = [
  {
    type: 'MRI Scanner',
    icon: Monitor,
    description: 'Magnetic Resonance Imaging Scanner',
    color: 'from-blue-400 to-blue-600'
  },
  {
    type: 'Ventilator',
    icon: Wind,
    description: 'Medical Ventilation Equipment',
    color: 'from-coral-400 to-coral-600'
  },
  {
    type: 'EEG Machine',
    icon: Brain,
    description: 'Electroencephalography Machine',
    color: 'from-purple-400 to-purple-600'
  }
] as const

type DeviceType = typeof deviceTypes[number]['type']

const DeviceSetup: React.FC = () => {
  const [selectedDevices, setSelectedDevices] = useState<Array<{
    type: DeviceType
    location: string
  }>>([])
  const [currentDevice, setCurrentDevice] = useState<DeviceType | null>(null)
  const [location, setLocation] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const { user, userProfile } = useAuth()
  const navigate = useNavigate()

  const handleDeviceSelect = (type: DeviceType) => {
    setCurrentDevice(type)
    setLocation('')
  }

  const addDevice = () => {
    if (!currentDevice || !location.trim()) return

    setSelectedDevices(prev => [...prev, {
      type: currentDevice,
      location: location.trim()
    }])
    setCurrentDevice(null)
    setLocation('')
  }

  const removeDevice = (index: number) => {
    setSelectedDevices(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async () => {
    if (selectedDevices.length === 0) {
      setError('Please add at least one device')
      return
    }

    if (!userProfile || !userProfile.hospital_id) {
      setError('User profile not loaded. Please try again.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const devices = selectedDevices.map(device => ({
        hospital_id: userProfile.hospital_id,
        device_type: device.type,
        location: device.location,
        status: 'OK' as const,
        technician_id: user!.id,
        next_maintenance_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days from now
        last_maintenance_date: new Date().toISOString()
      }))

      const { error: insertError } = await supabase
        .from('devices')
        .insert(devices)

      if (insertError) throw insertError

      navigate('/dashboard')
    } catch (error: any) {
      setError(error.message || 'Failed to set up devices')
    } finally {
      setLoading(false)
    }
  }

  // Show loading state while user profile is loading
  if (!userProfile) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-teal-50 to-blue-50 flex items-center justify-center p-4">
        <div className="max-w-4xl w-full">
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="text-center">
              <div className="mx-auto w-16 h-16 bg-teal-100 rounded-xl flex items-center justify-center mb-4">
                <Activity className="w-8 h-8 text-teal-600" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">Device Setup</h1>
              <p className="text-gray-600 mt-2">Loading your profile...</p>
              <div className="mt-4 flex justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-600"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 to-blue-50 flex items-center justify-center p-4">
      <div className="max-w-4xl w-full">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="mx-auto w-16 h-16 bg-teal-100 rounded-xl flex items-center justify-center mb-4">
              <Activity className="w-8 h-8 text-teal-600" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">Device Setup</h1>
            <p className="text-gray-600 mt-2">
              Select the medical devices you want to monitor at {userProfile?.hospitals?.name}
            </p>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm mb-6">
              {error}
            </div>
          )}

          {/* Device Selection */}
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            {deviceTypes.map((device) => {
              const Icon = device.icon
              const isSelected = currentDevice === device.type

              return (
                <button
                  key={device.type}
                  onClick={() => handleDeviceSelect(device.type)}
                  className={`p-6 rounded-xl border-2 transition-all text-left ${
                    isSelected
                      ? 'border-teal-500 bg-teal-50 shadow-lg scale-105'
                      : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                  }`}
                >
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${device.color} flex items-center justify-center mb-4`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-2">{device.type}</h3>
                  <p className="text-sm text-gray-600">{device.description}</p>
                </button>
              )
            })}
          </div>

          {/* Location Input */}
          {currentDevice && (
            <div className="bg-gray-50 rounded-xl p-6 mb-6">
              <h3 className="font-semibold text-gray-900 mb-4">
                Add {currentDevice} Location
              </h3>
              <div className="flex space-x-3">
                <div className="flex-1">
                  <div className="relative">
                    <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type="text"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-colors"
                      placeholder="e.g., Wing A, Room 101"
                      onKeyPress={(e) => e.key === 'Enter' && addDevice()}
                    />
                  </div>
                </div>
                <button
                  onClick={addDevice}
                  disabled={!location.trim()}
                  className="px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
                >
                  <Plus className="w-5 h-5" />
                  <span>Add</span>
                </button>
              </div>
            </div>
          )}

          {/* Selected Devices */}
          {selectedDevices.length > 0 && (
            <div className="mb-8">
              <h3 className="font-semibold text-gray-900 mb-4">Selected Devices ({selectedDevices.length})</h3>
              <div className="space-y-3">
                {selectedDevices.map((device, index) => {
                  const deviceInfo = deviceTypes.find(d => d.type === device.type)!
                  const Icon = deviceInfo.icon

                  return (
                    <div key={index} className="flex items-center justify-between p-4 bg-white border border-gray-200 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${deviceInfo.color} flex items-center justify-center`}>
                          <Icon className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900">{device.type}</h4>
                          <p className="text-sm text-gray-600">📍 {device.location}</p>
                        </div>
                      </div>
                      <button
                        onClick={() => removeDevice(index)}
                        className="p-2 text-gray-400 hover:text-red-600 transition-colors"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Submit Button */}
          <div className="flex justify-center">
            <button
              onClick={handleSubmit}
              disabled={loading || selectedDevices.length === 0}
              className="px-8 py-3 bg-teal-600 text-white rounded-lg font-medium hover:bg-teal-700 focus:ring-4 focus:ring-teal-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Setting Up...</span>
                </div>
              ) : (
                `Complete Setup (${selectedDevices.length} devices)`
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DeviceSetup