import React, { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { 
  Brain, 
  Thermometer, 
  Zap, 
  Activity, 
  Gauge, 
  Droplets, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  ArrowLeft,
  Save,
  BarChart3
} from 'lucide-react'

interface EEGSensorReading {
  temperature: number
  voltage: number
  current: number
  vibration: number
  pressure: number
  humidity: number
  usageHours: number
}

interface EEGPrediction {
  condition: string
  failure: string
  cause: string
  solution: string
  timeToFailure: number
  probabilities: string
}

const EEGMaintenanceLog: React.FC = () => {
  const [sensorReading, setSensorReading] = useState<EEGSensorReading>({
    temperature: 45,
    voltage: 220,
    current: 7,
    vibration: 1.5,
    pressure: 95,
    humidity: 50,
    usageHours: 5000
  })
  const [prediction, setPrediction] = useState<EEGPrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [notes, setNotes] = useState('')

  const { user, userProfile } = useAuth()
  const navigate = useNavigate()
  const { deviceId } = useParams()

  const handleInputChange = (field: keyof EEGSensorReading, value: number) => {
    setSensorReading(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const validateInputs = (): boolean => {
    const ranges = {
      temperature: [30, 80],
      voltage: [180, 250],
      current: [0, 15],
      vibration: [0, 5],
      pressure: [80, 120],
      humidity: [20, 80],
      usageHours: [0, 10000]
    }

    for (const [field, [min, max]] of Object.entries(ranges)) {
      const value = sensorReading[field as keyof EEGSensorReading]
      if (value < min || value > max) {
        setError(`${field} must be between ${min} and ${max}`)
        return false
      }
    }
    return true
  }

  const makePrediction = async () => {
    if (!validateInputs()) return

    setLoading(true)
    setError('')

    try {
      // Call the Python backend API
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          temperature: sensorReading.temperature,
          voltage: sensorReading.voltage,
          current: sensorReading.current,
          vibration: sensorReading.vibration,
          pressure: sensorReading.pressure,
          humidity: sensorReading.humidity,
          usageHours: sensorReading.usageHours
        })
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`)
      }

      const result = await response.json()
      
      if (result.error) {
        throw new Error(result.error)
      }

      setPrediction(result)
    } catch (error: any) {
      console.error('Prediction error:', error)
      // Fallback to simulation if API is not available
      const mockPrediction = simulateEEGPrediction(sensorReading)
      setPrediction(mockPrediction)
      setError('API unavailable, using simulation mode. ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const simulateEEGPrediction = (reading: EEGSensorReading): EEGPrediction => {
    // Simple rule-based simulation (replace with actual API call)
    let condition = 'Good'
    let failure = 'No'
    let cause = 'None'
    let solution = 'Continue routine maintenance'
    let timeToFailure = 365

    if (reading.temperature > 65 || reading.voltage < 190 || reading.voltage > 240) {
      condition = 'Critical'
      failure = 'Yes'
      cause = reading.temperature > 65 ? 'Overheating' : 'Voltage Surge'
      solution = reading.temperature > 65 ? 'Check cooling system' : 'Stabilize power supply'
      timeToFailure = 5
    } else if (reading.vibration > 3 || reading.usageHours > 8000) {
      condition = 'Warning'
      failure = 'Yes'
      cause = reading.vibration > 3 ? 'Mechanical Wear' : 'Wear & Tear'
      solution = reading.vibration > 3 ? 'Inspect moving parts' : 'Replace worn components'
      timeToFailure = 30
    } else if (reading.usageHours > 5000) {
      condition = 'Average'
      timeToFailure = 180
    }

    return {
      condition,
      failure,
      cause,
      solution,
      timeToFailure,
      probabilities: `Good: 60%\nAverage: 25%\nWarning: 10%\nCritical: 5%`
    }
  }

  const handleSubmit = async () => {
    if (!prediction) {
      setError('Please make a prediction first')
      return
    }

    setLoading(true)
    setError('')

    try {
      const { error: insertError } = await supabase
        .from('maintenance_logs')
        .insert({
          device_id: deviceId,
          uploaded_by: user!.id,
          date_of_maintenance: new Date().toISOString(),
          notes: `EEG Maintenance Log - Condition: ${prediction.condition}, Failure: ${prediction.failure}, TTF: ${prediction.timeToFailure} days. ${notes}`,
          log_file_url: null
        })

      if (insertError) throw insertError

      navigate('/dashboard')
    } catch (error: any) {
      setError(error.message || 'Failed to save maintenance log')
    } finally {
      setLoading(false)
    }
  }

  const getConditionColor = (condition: string) => {
    switch (condition) {
      case 'Good': return 'text-green-600 bg-green-100'
      case 'Average': return 'text-blue-600 bg-blue-100'
      case 'Warning': return 'text-yellow-600 bg-yellow-100'
      case 'Critical': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getConditionIcon = (condition: string) => {
    switch (condition) {
      case 'Good': return CheckCircle
      case 'Average': return BarChart3
      case 'Warning': return AlertTriangle
      case 'Critical': return AlertTriangle
      default: return CheckCircle
    }
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={() => navigate('/dashboard')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 mb-4 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back to Dashboard</span>
        </button>
        
        <div className="flex items-center space-x-3 mb-2">
          <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
            <Brain className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">EEG Machine Maintenance Log</h1>
            <p className="text-gray-600">Enter sensor readings for failure prediction analysis</p>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Sensor Input Form */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-6">Sensor Readings</h2>
            
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm mb-6">
                {error}
              </div>
            )}

            <div className="space-y-4">
              {/* Temperature */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Thermometer className="w-4 h-4 inline mr-2" />
                  Temperature (°C)
                </label>
                <input
                  type="number"
                  value={sensorReading.temperature}
                  onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  min="30"
                  max="80"
                  step="0.1"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 30-80°C</p>
              </div>

              {/* Voltage */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Zap className="w-4 h-4 inline mr-2" />
                  Voltage (V)
                </label>
                <input
                  type="number"
                  value={sensorReading.voltage}
                  onChange={(e) => handleInputChange('voltage', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  min="180"
                  max="250"
                  step="0.1"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 180-250V</p>
              </div>

              {/* Current */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Activity className="w-4 h-4 inline mr-2" />
                  Current (A)
                </label>
                <input
                  type="number"
                  value={sensorReading.current}
                  onChange={(e) => handleInputChange('current', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  min="0"
                  max="15"
                  step="0.1"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 0-15A</p>
              </div>

              {/* Vibration */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Gauge className="w-4 h-4 inline mr-2" />
                  Vibration (g)
                </label>
                <input
                  type="number"
                  value={sensorReading.vibration}
                  onChange={(e) => handleInputChange('vibration', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  min="0"
                  max="5"
                  step="0.1"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 0-5g</p>
              </div>

              {/* Pressure */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Gauge className="w-4 h-4 inline mr-2" />
                  Pressure (kPa)
                </label>
                <input
                  type="number"
                  value={sensorReading.pressure}
                  onChange={(e) => handleInputChange('pressure', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  min="80"
                  max="120"
                  step="0.1"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 80-120kPa</p>
              </div>

              {/* Humidity */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Droplets className="w-4 h-4 inline mr-2" />
                  Humidity (%)
                </label>
                <input
                  type="number"
                  value={sensorReading.humidity}
                  onChange={(e) => handleInputChange('humidity', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  min="20"
                  max="80"
                  step="0.1"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 20-80%</p>
              </div>

              {/* Usage Hours */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Clock className="w-4 h-4 inline mr-2" />
                  Usage Hours (h)
                </label>
                <input
                  type="number"
                  value={sensorReading.usageHours}
                  onChange={(e) => handleInputChange('usageHours', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  min="0"
                  max="10000"
                  step="1"
                />
                <p className="text-xs text-gray-500 mt-1">Range: 0-10,000h</p>
              </div>
            </div>

            <button
              onClick={makePrediction}
              disabled={loading}
              className="w-full mt-6 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <BarChart3 className="w-5 h-5" />
                  <span>Analyze & Predict</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Prediction Results */}
        <div className="space-y-6">
          {prediction && (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-6">Prediction Results</h2>
              
              <div className="space-y-4">
                {/* Condition */}
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="font-medium text-gray-700">Condition:</span>
                  <span className={`inline-flex items-center space-x-1 px-3 py-1 rounded-full text-sm font-medium ${getConditionColor(prediction.condition)}`}>
                    {React.createElement(getConditionIcon(prediction.condition), { className: 'w-4 h-4' })}
                    <span>{prediction.condition}</span>
                  </span>
                </div>

                {/* Failure Status */}
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="font-medium text-gray-700">Failure Predicted:</span>
                  <span className={`font-semibold ${prediction.failure === 'Yes' ? 'text-red-600' : 'text-green-600'}`}>
                    {prediction.failure}
                  </span>
                </div>

                {/* Time to Failure */}
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="font-medium text-gray-700">Time to Failure:</span>
                  <span className="font-semibold text-blue-600">
                    {prediction.timeToFailure} days
                  </span>
                </div>

                {/* Root Cause */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <span className="font-medium text-gray-700 block mb-2">Root Cause:</span>
                  <span className="text-gray-900">{prediction.cause}</span>
                </div>

                {/* Recommended Solution */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <span className="font-medium text-gray-700 block mb-2">Recommended Solution:</span>
                  <span className="text-gray-900">{prediction.solution}</span>
                </div>

                {/* Probabilities */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <span className="font-medium text-gray-700 block mb-2">Class Probabilities:</span>
                  <pre className="text-sm text-gray-900 whitespace-pre-line">{prediction.probabilities}</pre>
                </div>
              </div>
            </div>
          )}

          {/* Notes Section */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Additional Notes</h2>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add any additional observations or notes about the maintenance..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 h-24 resize-none"
            />
          </div>

          {/* Submit Button */}
          {prediction && (
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Saving Log...</span>
                </>
              ) : (
                <>
                  <Save className="w-5 h-5" />
                  <span>Save Maintenance Log</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default EEGMaintenanceLog
