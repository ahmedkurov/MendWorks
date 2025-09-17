import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { 
  Monitor, 
  Wind, 
  Brain,
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  MapPin,
  Phone,
  Plus,
  Filter,
  Search,
  TrendingUp,
  Calendar,
  Activity,
  Trash2,
  Bot,
  X,
  AlertCircle
} from 'lucide-react'
import { format, isAfter, isBefore, addDays } from 'date-fns'

interface Device {
  id: string
  device_type: 'MRI Scanner' | 'Ventilator' | 'EEG Machine'
  location: string
  status: 'OK' | 'Warning' | 'Danger'
  next_maintenance_date: string
  last_maintenance_date: string
  created_at: string
  maintenance_logs?: Array<{
    id: string
    notes: string
    date_of_maintenance: string
  }>
}

const Dashboard: React.FC = () => {
  const [devices, setDevices] = useState<Device[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'OK' | 'Warning' | 'Danger'>('all')
  const [sortBy, setSortBy] = useState<'status' | 'maintenance' | 'location'>('status')
  const [showCriticalAlert, setShowCriticalAlert] = useState(false)
  const [criticalDevices, setCriticalDevices] = useState<Device[]>([])

  const auth = useAuth()
  const userProfile = auth.userProfile
  const profileLoading = auth.profileLoading

  useEffect(() => {
    fetchDevices()
  }, [userProfile?.hospital_id])

  const fetchDevices = async () => {
    try {
      setError(null)
      setLoading(true)
      
      if (!userProfile?.hospital_id) {
        console.log('No hospital_id found in userProfile:', userProfile)
        setLoading(false)
        return
      }

      console.log('Fetching devices for hospital_id:', userProfile.hospital_id)

      const { data, error } = await supabase
        .from('devices')
        .select(`
          *,
          maintenance_logs (
            id,
            notes,
            date_of_maintenance
          )
        `)
        .eq('hospital_id', userProfile.hospital_id)
        .order('created_at', { ascending: false })

      if (error) {
        console.error('Supabase error:', error)
        setError(`Database error: ${error.message}`)
        setLoading(false)
        return
      }

      console.log('Fetched devices:', data)

      // Get AI predictions for all device types
      const eegDevices = data.filter(device => device.device_type === 'EEG Machine')
      const mriDevices = data.filter(device => device.device_type === 'MRI Scanner')
      const ventilatorDevices = data.filter(device => device.device_type === 'Ventilator')
      
      let aiPredictions = {}
      
      // Fetch EEG predictions
      if (eegDevices.length > 0) {
        const deviceIds = eegDevices.map(d => d.id)
        const { data: predictions } = await supabase
          .from('eeg_predictions')
          .select('device_id, predicted_condition, confidence_score, prediction_date, time_to_failure_days')
          .in('device_id', deviceIds)
          .order('prediction_date', { ascending: false })
        
        predictions?.forEach(prediction => {
          if (!aiPredictions[prediction.device_id]) {
            aiPredictions[prediction.device_id] = { ...prediction, type: 'eeg' }
          }
        })
      }
      
      // Fetch MRI predictions
      if (mriDevices.length > 0) {
        const deviceIds = mriDevices.map(d => d.id)
        const { data: predictions } = await supabase
          .from('mri_predictions')
          .select('device_id, predicted_condition, confidence_score, prediction_date, time_to_failure_days')
          .in('device_id', deviceIds)
          .order('prediction_date', { ascending: false })
        
        predictions?.forEach(prediction => {
          if (!aiPredictions[prediction.device_id]) {
            aiPredictions[prediction.device_id] = { ...prediction, type: 'mri' }
          }
        })
      }
      
      // Fetch Ventilator predictions
      if (ventilatorDevices.length > 0) {
        const deviceIds = ventilatorDevices.map(d => d.id)
        const { data: predictions } = await supabase
          .from('ventilator_predictions')
          .select('device_id, predicted_condition, confidence_score, prediction_date, time_to_failure_days')
          .in('device_id', deviceIds)
          .order('prediction_date', { ascending: false })
        
        predictions?.forEach(prediction => {
          if (!aiPredictions[prediction.device_id]) {
            aiPredictions[prediction.device_id] = { ...prediction, type: 'ventilator' }
          }
        })
      }

      // Update device statuses based on maintenance dates and AI predictions
      const updatedDevices = data.map(device => {
        // For all device types with AI predictions, update status based on prediction
        if (aiPredictions[device.id]) {
          const prediction = aiPredictions[device.id]
          const aiCondition = prediction.predicted_condition
          let aiStatus: 'OK' | 'Warning' | 'Danger' = 'OK'
          
          switch (aiCondition) {
            case 'Good':
            case 'Average':
              aiStatus = 'OK'
              break
            case 'Warning':
              aiStatus = 'Warning'
              break
            case 'Critical':
              aiStatus = 'Danger'
              break
            default:
              aiStatus = 'OK'
          }
          
          return { 
            ...device, 
            status: aiStatus,
            aiCondition: aiCondition,
            hasAIData: true,
            aiConfidence: prediction.confidence_score
          }
        }
        
        // For non-EEG devices or EEG devices without AI data, use maintenance-based status
        let status: 'OK' | 'Warning' | 'Danger' = 'OK'
        
        if (device.next_maintenance_date) {
          const nextMaintenance = new Date(device.next_maintenance_date)
          const now = new Date()
          const warningThreshold = addDays(now, 7) // 7 days warning
          
          if (!isNaN(nextMaintenance.getTime())) {
            if (isBefore(nextMaintenance, now)) {
              status = 'Danger' // Overdue
            } else if (isBefore(nextMaintenance, warningThreshold)) {
              status = 'Warning' // Due soon
            }
          }
        }

        return { ...device, status }
      })

      setDevices(updatedDevices)
      
      // Check for critical devices and show alert
      checkCriticalDevices(updatedDevices)
    } catch (error) {
      console.error('Error fetching devices:', error)
      setError(`Failed to load devices: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const checkCriticalDevices = (devices: Device[]) => {
    const critical = devices.filter(device => device.status === 'Danger')
    if (critical.length > 0) {
      setCriticalDevices(critical)
      setShowCriticalAlert(true)
    }
  }

  const deleteDevice = async (deviceId: string) => {
    try {
      if (!userProfile?.id) {
        setError('User not authenticated')
        return
      }

      // Confirm deletion
      const confirmed = window.confirm(
        'Are you sure you want to delete this device? This will permanently remove the device and all its related data (maintenance logs, AI predictions, etc.). This action cannot be undone.'
      )
      
      if (!confirmed) return

      setLoading(true)
      setError(null)

      // Call the backend API to delete the device
      const response = await fetch('http://localhost:5000/api/delete-device', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          device_id: deviceId,
          user_id: userProfile.id
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to delete device')
      }

      // Refresh the devices list
      await fetchDevices()
      
      console.log('Device deleted successfully')
    } catch (error) {
      console.error('Error deleting device:', error)
      setError(`Failed to delete device: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'MRI Scanner':
        return Monitor
      case 'Ventilator':
        return Wind
      case 'EEG Machine':
        return Brain
      default:
        return Monitor
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'OK':
        return 'text-green-600 bg-green-100'
      case 'Warning':
        return 'text-yellow-600 bg-yellow-100'
      case 'Danger':
        return 'text-red-600 bg-red-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'OK':
        return CheckCircle
      case 'Warning':
        return Clock
      case 'Danger':
        return AlertTriangle
      default:
        return CheckCircle
    }
  }

  const getLatestAIPrediction = (device: Device) => {
    // Check if device has AI data for any device type
    if (!device.hasAIData) return null
    
    return {
      predicted_condition: device.aiCondition,
      predicted_failure: 'No', // We can add this if needed
      time_to_failure_days: null, // We can add this if needed
      prediction_date: new Date().toISOString(),
      confidence_score: device.aiConfidence
    }
  }

  const filteredAndSortedDevices = devices
    .filter(device => 
      (statusFilter === 'all' || device.status === statusFilter) &&
      (device.device_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
       device.location.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'status':
          const statusOrder = { 'Danger': 0, 'Warning': 1, 'OK': 2 }
          return statusOrder[a.status] - statusOrder[b.status]
        case 'maintenance':
          return new Date(a.next_maintenance_date).getTime() - new Date(b.next_maintenance_date).getTime()
        case 'location':
          return a.location.localeCompare(b.location)
        default:
          return 0
      }
    })

  const statusCounts = {
    total: devices.length,
    ok: devices.filter(d => d.status === 'OK').length,
    warning: devices.filter(d => d.status === 'Warning').length,
    danger: devices.filter(d => d.status === 'Danger').length
  }

  // Count devices with AI analysis
  const devicesWithAI = devices.filter(d => 
    d.maintenance_logs?.some(log => 
      log.notes.includes('EEG Maintenance Log') || 
      log.notes.includes('MRI Maintenance Log') || 
      log.notes.includes('Ventilator Maintenance Log')
    )
  ).length

  // Debug information
  console.log('Dashboard render - loading:', loading, 'userProfile:', userProfile, 'devices:', devices)

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-32 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // Show loading state if profile is still loading
  if (profileLoading) {
    return (
      <div className="p-6">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-blue-800 mb-2">Loading Profile...</h2>
          <p className="text-blue-700 mb-4">
            Please wait while we load your profile information.
          </p>
          <div className="animate-pulse">
            <div className="h-4 bg-blue-200 rounded w-1/4 mb-2"></div>
            <div className="h-4 bg-blue-200 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    )
  }

  // Show debug info if no user profile
  if (!userProfile) {
    return (
      <div className="p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-yellow-800 mb-2">Authentication Required</h2>
          <p className="text-yellow-700 mb-4">
            Please log in to view the dashboard. User profile: {JSON.stringify(userProfile)}
          </p>
          <Link
            to="/login"
            className="inline-flex items-center px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
          >
            Go to Login
          </Link>
        </div>
      </div>
    )
  }

  // Add error boundary
  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-red-800 mb-2">Error Loading Dashboard</h2>
          <p className="text-red-700 mb-4">{error}</p>
          <button
            onClick={() => {
              setError(null)
              setLoading(true)
              fetchDevices()
            }}
            className="inline-flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Critical Device Alert */}
      {showCriticalAlert && criticalDevices.length > 0 && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
                    <AlertCircle className="w-6 h-6 text-red-600" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-red-800">Critical Device Alert</h2>
                    <p className="text-red-600">Immediate attention required</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowCriticalAlert(false)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
              
              <div className="space-y-4">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h3 className="font-semibold text-red-800 mb-2">
                    {criticalDevices.length} device{criticalDevices.length > 1 ? 's' : ''} require{criticalDevices.length === 1 ? 's' : ''} immediate attention
                  </h3>
                  <p className="text-red-700 text-sm">
                    The following devices are in critical condition and may require immediate maintenance or replacement.
                  </p>
                </div>
                
                <div className="space-y-3">
                  {criticalDevices.map((device) => (
                    <div key={device.id} className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                            {device.device_type === 'EEG Machine' && <Brain className="w-5 h-5 text-red-600" />}
                            {device.device_type === 'MRI Scanner' && <Monitor className="w-5 h-5 text-red-600" />}
                            {device.device_type === 'Ventilator' && <Wind className="w-5 h-5 text-red-600" />}
                          </div>
                          <div>
                            <h4 className="font-semibold text-gray-900">{device.device_type}</h4>
                            <p className="text-sm text-gray-600">Location: {device.location}</p>
                            {device.aiCondition && (
                              <p className="text-sm text-red-600 font-medium">
                                AI Status: {device.aiCondition} 
                                {device.aiConfidence && ` (${Math.round(device.aiConfidence * 100)}% confidence)`}
                              </p>
                            )}
                          </div>
                        </div>
                        <div className="text-right">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                            Critical
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="flex justify-end space-x-3 pt-4">
                  <button
                    onClick={() => setShowCriticalAlert(false)}
                    className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                  >
                    Dismiss
                  </button>
                  <button
                    onClick={() => {
                      setShowCriticalAlert(false)
                      // You could add navigation to maintenance logs here
                    }}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    View Maintenance Logs
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Equipment Dashboard</h1>
          <p className="text-gray-600 mt-1">{userProfile?.hospitals?.name}</p>
        </div>
        <div className="mt-4 md:mt-0 flex space-x-3">
          <Link
            to="/add-device"
            className="inline-flex items-center space-x-2 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>Add Device</span>
          </Link>
        </div>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-xl border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Devices</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{statusCounts.total}</p>
              {devicesWithAI > 0 && (
                <p className="text-xs text-blue-600 mt-1">
                  {devicesWithAI} with AI analysis
                </p>
              )}
            </div>
            <TrendingUp className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Healthy</p>
              <p className="text-2xl font-bold text-green-600 mt-1">{statusCounts.ok}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Warning</p>
              <p className="text-2xl font-bold text-yellow-600 mt-1">{statusCounts.warning}</p>
            </div>
            <Clock className="w-8 h-8 text-yellow-500" />
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Critical</p>
              <p className="text-2xl font-bold text-red-600 mt-1">{statusCounts.danger}</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-500" />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-xl border border-gray-200">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search devices or locations..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500"
              />
            </div>
          </div>

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as any)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500"
          >
            <option value="all">All Status</option>
            <option value="OK">Healthy</option>
            <option value="Warning">Warning</option>
            <option value="Danger">Critical</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500"
          >
            <option value="status">Sort by Status</option>
            <option value="maintenance">Sort by Maintenance</option>
            <option value="location">Sort by Location</option>
          </select>
        </div>
      </div>

      {/* Device List */}
      {filteredAndSortedDevices.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
          <Monitor className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No devices found</h3>
          <p className="text-gray-600 mb-6">
            {devices.length === 0 
              ? "Get started by adding your first medical device to monitor."
              : "Try adjusting your search or filter criteria."
            }
          </p>
          {devices.length === 0 && (
            <Link
              to="/add-device"
              className="inline-flex items-center space-x-2 px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors"
            >
              <Plus className="w-5 h-5" />
              <span>Add Your First Device</span>
            </Link>
          )}
        </div>
      ) : (
        <div className="grid gap-6">
          {filteredAndSortedDevices.map((device) => {
            const DeviceIcon = getDeviceIcon(device.device_type)
            const StatusIcon = getStatusIcon(device.status)
            const nextMaintenance = device.next_maintenance_date ? new Date(device.next_maintenance_date) : null
            const lastMaintenance = device.last_maintenance_date ? new Date(device.last_maintenance_date) : null
            const aiPrediction = getLatestAIPrediction(device)

            return (
              <div key={device.id} className="bg-white rounded-xl border border-gray-200 hover:shadow-lg transition-shadow">
                <div className="p-6">
                  <div className="flex flex-col md:flex-row md:items-center justify-between">
                    <div className="flex items-start space-x-4 mb-4 md:mb-0">
                      <div className="w-12 h-12 bg-teal-100 rounded-xl flex items-center justify-center flex-shrink-0">
                        <DeviceIcon className="w-6 h-6 text-teal-600" />
                      </div>
                      
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <h3 className="text-lg font-semibold text-gray-900">{device.device_type}</h3>
                          <span className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(device.status)}`}>
                            <StatusIcon className="w-3 h-3" />
                            <span>{device.status}</span>
                          </span>
                        </div>
                        
                        <div className="flex items-center text-sm text-gray-600 mb-1">
                          <MapPin className="w-4 h-4 mr-1" />
                          <span>{device.location}</span>
                        </div>
                        
                        <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-6 text-sm text-gray-600">
                          <div className="flex items-center">
                            <Calendar className="w-4 h-4 mr-1" />
                            <span>Next: {nextMaintenance && !isNaN(nextMaintenance.getTime()) ? format(nextMaintenance, 'MMM dd, yyyy') : 'Not set'}</span>
                          </div>
                          <div className="flex items-center mt-1 sm:mt-0">
                            <span>Last: {lastMaintenance && !isNaN(lastMaintenance.getTime()) ? format(lastMaintenance, 'MMM dd, yyyy') : 'Not set'}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      {device.device_type === 'EEG Machine' ? (
                        <Link
                          to={`/device/${device.id}/eeg-maintenance`}
                          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium flex items-center space-x-1"
                        >
                          <Brain className="w-4 h-4" />
                          <span>AI Analysis</span>
                        </Link>
                      ) : device.device_type === 'MRI Scanner' ? (
                        <Link
                          to={`/device/${device.id}/mri-maintenance`}
                          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium flex items-center space-x-1"
                        >
                          <Monitor className="w-4 h-4" />
                          <span>AI Analysis</span>
                        </Link>
                      ) : device.device_type === 'Ventilator' ? (
                        <Link
                          to={`/device/${device.id}/ventilator-maintenance`}
                          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium flex items-center space-x-1"
                        >
                          <Wind className="w-4 h-4" />
                          <span>AI Analysis</span>
                        </Link>
                      ) : (
                        <Link
                          to={`/device/${device.id}/maintenance`}
                          className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors text-sm font-medium"
                        >
                          Add Log
                        </Link>
                      )}
                      {/* Temporarily disabled delete functionality */}
                      {/* <button
                        onClick={() => deleteDevice(device.id)}
                        className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm font-medium flex items-center space-x-1"
                        title="Delete Device"
                      >
                        <Trash2 className="w-4 h-4" />
                        <span>Delete</span>
                      </button> */}
                    </div>
                  </div>
                </div>

                {/* Risk Assessment */}
                <div className="px-6 pb-4">
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-gray-700">
                        {device.device_type === 'EEG Machine' && aiPrediction ? 'AI Risk Assessment:' : 'Failure Risk Assessment:'}
                      </span>
                      <span className={`font-semibold ${
                        device.status === 'OK' ? 'text-green-600' : 
                        device.status === 'Warning' ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {device.status === 'OK' ? 'Low Risk' : 
                         device.status === 'Warning' ? 'Medium Risk' : 'High Risk'}
                      </span>
                    </div>
                    <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          device.status === 'OK' ? 'bg-green-500 w-1/4' : 
                          device.status === 'Warning' ? 'bg-yellow-500 w-2/3' : 'bg-red-500 w-full'
                        }`}
                      ></div>
                    </div>
                    {aiPrediction && (
                      <div className="mt-2 text-xs text-gray-600">
                        Based on AI analysis: {aiPrediction.predicted_condition} condition
                      </div>
                    )}
                  </div>
                  
                  {/* AI Prediction Display for all device types */}
                  {aiPrediction && (
                    <div className={`mt-3 rounded-lg p-3 border ${
                      device.device_type === 'EEG Machine' ? 'bg-purple-50 border-purple-200' :
                      device.device_type === 'MRI Scanner' ? 'bg-blue-50 border-blue-200' :
                      device.device_type === 'Ventilator' ? 'bg-green-50 border-green-200' :
                      'bg-gray-50 border-gray-200'
                    }`}>
                      <div className="flex items-center space-x-2 mb-2">
                        {device.device_type === 'EEG Machine' ? <Brain className="w-4 h-4 text-purple-600" /> :
                         device.device_type === 'MRI Scanner' ? <Monitor className="w-4 h-4 text-blue-600" /> :
                         device.device_type === 'Ventilator' ? <Wind className="w-4 h-4 text-green-600" /> :
                         <Activity className="w-4 h-4 text-gray-600" />}
                        <span className={`font-medium text-sm ${
                          device.device_type === 'EEG Machine' ? 'text-purple-800' :
                          device.device_type === 'MRI Scanner' ? 'text-blue-800' :
                          device.device_type === 'Ventilator' ? 'text-green-800' :
                          'text-gray-800'
                        }`}>
                          Latest AI Analysis
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-600">Condition:</span>
                          <span className={`ml-1 font-medium ${
                            aiPrediction.predicted_condition === 'Good' ? 'text-green-600' :
                            aiPrediction.predicted_condition === 'Average' ? 'text-blue-600' :
                            aiPrediction.predicted_condition === 'Warning' ? 'text-yellow-600' :
                            'text-red-600'
                          }`}>
                            {aiPrediction.predicted_condition}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">TTF:</span>
                          <span className="ml-1 font-medium text-blue-600">
                            {aiPrediction.time_to_failure_days} days
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">Confidence:</span>
                          <span className="ml-1 font-medium text-green-600">
                            {aiPrediction.confidence_score?.toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">Date:</span>
                          <span className="ml-1 font-medium text-gray-700">
                            {aiPrediction.prediction_date && !isNaN(new Date(aiPrediction.prediction_date).getTime()) 
                              ? format(new Date(aiPrediction.prediction_date), 'MMM dd') 
                              : 'N/A'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Emergency Contact */}
      <div className="bg-gradient-to-r from-coral-50 to-red-50 rounded-xl border border-coral-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Emergency Maintenance Support</h3>
            <p className="text-gray-600 mb-4">Need immediate assistance with critical equipment?</p>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-sm text-gray-700">
                <Phone className="w-4 h-4 mr-2" />
                <span>{userProfile?.hospitals?.contact_number || '(555) 123-4567'}</span>
              </div>
              <div className="flex items-center text-sm text-gray-700">
                <span>24/7 Available</span>
              </div>
            </div>
          </div>
          <a
            href={`tel:${userProfile?.hospitals?.contact_number || '5551234567'}`}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium flex items-center space-x-2"
          >
            <Phone className="w-5 h-5" />
            <span>Call Now</span>
          </a>
        </div>
      </div>
    </div>
  )
}

export default Dashboard