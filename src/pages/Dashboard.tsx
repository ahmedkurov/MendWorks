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
  Calendar
} from 'lucide-react'
import { format, isAfter, isBefore, addDays } from 'date-fns'

interface Device {
  id: string
  device_type: 'MRI Scanner' | 'CT Scanner' | 'Ventilator' | 'EEG Machine'
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
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'OK' | 'Warning' | 'Danger'>('all')
  const [sortBy, setSortBy] = useState<'status' | 'maintenance' | 'location'>('status')

  const { userProfile } = useAuth()

  useEffect(() => {
    fetchDevices()
  }, [userProfile?.hospital_id])

  const fetchDevices = async () => {
    try {
      if (!userProfile?.hospital_id) return

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

      if (error) throw error

      // Get AI predictions for EEG devices
      const eegDevices = data.filter(device => device.device_type === 'EEG Machine')
      let aiPredictions = {}
      
      if (eegDevices.length > 0) {
        const deviceIds = eegDevices.map(d => d.id)
        const { data: predictions } = await supabase
          .from('eeg_predictions')
          .select('device_id, predicted_condition, confidence_score, prediction_date')
          .in('device_id', deviceIds)
          .order('prediction_date', { ascending: false })
        
        // Group predictions by device_id and get the latest for each device
        predictions?.forEach(prediction => {
          if (!aiPredictions[prediction.device_id]) {
            aiPredictions[prediction.device_id] = prediction
          }
        })
      }

      // Update device statuses based on maintenance dates and AI predictions
      const updatedDevices = data.map(device => {
        // For EEG machines, check if we have AI prediction data
        if (device.device_type === 'EEG Machine' && aiPredictions[device.id]) {
          const prediction = aiPredictions[device.id]
          const aiCondition = prediction.predicted_condition
          let aiStatus: 'OK' | 'Warning' | 'Danger' = 'OK'
          
          switch (aiCondition) {
            case 'Normal':
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
        const nextMaintenance = new Date(device.next_maintenance_date)
        const now = new Date()
        const warningThreshold = addDays(now, 7) // 7 days warning
        
        let status: 'OK' | 'Warning' | 'Danger' = 'OK'
        
        if (isBefore(nextMaintenance, now)) {
          status = 'Danger' // Overdue
        } else if (isBefore(nextMaintenance, warningThreshold)) {
          status = 'Warning' // Due soon
        }

        return { ...device, status }
      })

      setDevices(updatedDevices)
    } catch (error) {
      console.error('Error fetching devices:', error)
    } finally {
      setLoading(false)
    }
  }

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'MRI Scanner':
      case 'CT Scanner':
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
    if (device.device_type !== 'EEG Machine' || !device.hasAIData) return null
    
    return {
      condition: device.aiCondition,
      failure: 'No', // We can add this if needed
      timeToFailure: null, // We can add this if needed
      date: new Date().toISOString(),
      confidence: device.aiConfidence
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

  // Count EEG devices with AI analysis
  const eegDevicesWithAI = devices.filter(d => 
    d.device_type === 'EEG Machine' && 
    d.maintenance_logs?.some(log => log.notes.includes('EEG Maintenance Log'))
  ).length

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

  return (
    <div className="p-6 space-y-6">
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
              {eegDevicesWithAI > 0 && (
                <p className="text-xs text-purple-600 mt-1">
                  {eegDevicesWithAI} with AI analysis
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
            const nextMaintenance = new Date(device.next_maintenance_date)
            const lastMaintenance = new Date(device.last_maintenance_date)
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
                            <span>Next: {format(nextMaintenance, 'MMM dd, yyyy')}</span>
                          </div>
                          <div className="flex items-center mt-1 sm:mt-0">
                            <span>Last: {format(lastMaintenance, 'MMM dd, yyyy')}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <Link
                        to={`/device/${device.id}`}
                        className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium"
                      >
                        View Details
                      </Link>
                      {device.device_type === 'EEG Machine' ? (
                        <Link
                          to={`/device/${device.id}/eeg-maintenance`}
                          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium flex items-center space-x-1"
                        >
                          <Brain className="w-4 h-4" />
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
                    {device.device_type === 'EEG Machine' && aiPrediction && (
                      <div className="mt-2 text-xs text-gray-600">
                        Based on AI analysis: {aiPrediction.condition} condition
                      </div>
                    )}
                  </div>
                  
                  {/* AI Prediction Display for EEG Machines */}
                  {device.device_type === 'EEG Machine' && aiPrediction && (
                    <div className="mt-3 bg-purple-50 rounded-lg p-3 border border-purple-200">
                      <div className="flex items-center space-x-2 mb-2">
                        <Brain className="w-4 h-4 text-purple-600" />
                        <span className="font-medium text-purple-800 text-sm">Latest AI Analysis</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-600">Condition:</span>
                          <span className={`ml-1 font-medium ${
                            aiPrediction.condition === 'Good' ? 'text-green-600' :
                            aiPrediction.condition === 'Average' ? 'text-blue-600' :
                            aiPrediction.condition === 'Warning' ? 'text-yellow-600' :
                            'text-red-600'
                          }`}>
                            {aiPrediction.condition}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">TTF:</span>
                          <span className="ml-1 font-medium text-blue-600">
                            {aiPrediction.timeToFailure} days
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">Failure:</span>
                          <span className={`ml-1 font-medium ${
                            aiPrediction.failure === 'No' ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {aiPrediction.failure}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">Date:</span>
                          <span className="ml-1 font-medium text-gray-700">
                            {format(new Date(aiPrediction.date), 'MMM dd')}
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