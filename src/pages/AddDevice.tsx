import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { Monitor, Wind, MapPin, ArrowLeft, Plus, Brain } from 'lucide-react'

const deviceTypes = [
  {
    type: 'MRI Scanner',
    icon: Monitor,
    description: 'Magnetic Resonance Imaging Scanner',
    color: 'from-blue-400 to-blue-600'
  },
  {
    type: 'CT Scanner',
    icon: Monitor,
    description: 'Computed Tomography Scanner',
    color: 'from-teal-400 to-teal-600'
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

const AddDevice: React.FC = () => {
  const [selectedType, setSelectedType] = useState<DeviceType | null>(null)
  const [location, setLocation] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const { user, userProfile } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!selectedType || !location.trim()) {
      setError('Please select a device type and enter a location')
      return
    }

    setLoading(true)
    setError('')

    try {
      const { error: insertError } = await supabase
        .from('devices')
        .insert({
          hospital_id: userProfile.hospital_id,
          device_type: selectedType,
          location: location.trim(),
          status: 'OK' as const,
          technician_id: user!.id,
          next_maintenance_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days from now
          last_maintenance_date: new Date().toISOString()
        })

      if (insertError) throw insertError

      navigate('/dashboard')
    } catch (error: any) {
      setError(error.message || 'Failed to add device')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={() => navigate('/dashboard')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 mb-4 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back to Dashboard</span>
        </button>
        
        <h1 className="text-2xl font-bold text-gray-900">Add New Device</h1>
        <p className="text-gray-600 mt-1">Add a medical device to your monitoring system</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        {/* Device Type Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-4">
            Select Device Type *
          </label>
          <div className="grid md:grid-cols-3 gap-4">
            {deviceTypes.map((device) => {
              const Icon = device.icon
              const isSelected = selectedType === device.type

              return (
                <button
                  key={device.type}
                  type="button"
                  onClick={() => setSelectedType(device.type)}
                  className={`p-6 rounded-xl border-2 transition-all text-left ${
                    isSelected
                      ? 'border-teal-500 bg-teal-50 shadow-lg'
                      : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                  }`}
                >
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${device.color} flex items-center justify-center mb-4`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-2">{device.type}</h3>
                  <p className="text-sm text-gray-600">{device.description}</p>
                  {isSelected && (
                    <div className="mt-3 flex items-center text-teal-600 text-sm font-medium">
                      <div className="w-4 h-4 bg-teal-600 rounded-full mr-2 flex items-center justify-center">
                        <div className="w-2 h-2 bg-white rounded-full"></div>
                      </div>
                      Selected
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        </div>

        {/* Location Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Device Location *
          </label>
          <div className="relative">
            <MapPin className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-colors"
              placeholder="e.g., Wing A, Room 101, ICU Bay 3"
              required
            />
          </div>
          <p className="text-sm text-gray-500 mt-2">
            Specify the exact location where this device is installed
          </p>
        </div>

        {/* Device Preview */}
        {selectedType && location && (
          <div className="bg-gray-50 rounded-xl p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Device Preview</h3>
            <div className="flex items-center space-x-4">
              <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${deviceTypes.find(d => d.type === selectedType)?.color} flex items-center justify-center`}>
                {React.createElement(deviceTypes.find(d => d.type === selectedType)?.icon!, { className: 'w-6 h-6 text-white' })}
              </div>
              <div>
                <h4 className="font-medium text-gray-900">{selectedType}</h4>
                <p className="text-sm text-gray-600 flex items-center">
                  <MapPin className="w-4 h-4 mr-1" />
                  {location}
                </p>
                <div className="flex items-center mt-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  <span className="text-sm text-green-600 font-medium">Status: OK</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-end space-x-3">
          <button
            type="button"
            onClick={() => navigate('/dashboard')}
            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading || !selectedType || !location.trim()}
            className="px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 focus:ring-4 focus:ring-teal-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Adding Device...</span>
              </>
            ) : (
              <>
                <Plus className="w-5 h-5" />
                <span>Add Device</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}

export default AddDevice