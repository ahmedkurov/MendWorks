import React, { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { 
  Wind, 
  Thermometer, 
  Gauge, 
  Activity, 
  Droplets, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  ArrowLeft,
  Save,
  BarChart3,
  Battery,
  Zap,
  Cpu
} from 'lucide-react'

interface VentilatorSensorReading {
  airway_pressure: number
  tidal_volume: number
  respiratory_rate: number
  oxygen_concentration: number
  inspiratory_flow: number
  exhaled_co2: number
  humidifier_temperature: number
  compressor_pressure: number
  valve_response: number
  battery_level: number
  power_consumption: number
  alarm_frequency: number
  system_runtime: number
  vibration: number
  internal_temperature: number
  filter_status: number
}

interface VentilatorPrediction {
  condition: string
  failure: string
  cause: string
  solution: string
  time_to_failure: number
  confidence: number
  probabilities: string
}

const VentilatorMaintenanceLog: React.FC = () => {
  const { deviceId } = useParams<{ deviceId: string }>()
  const navigate = useNavigate()
  const { user } = useAuth()
  
  const [sensorReading, setSensorReading] = useState<VentilatorSensorReading>({
    airway_pressure: 25.0,
    tidal_volume: 500.0,
    respiratory_rate: 15.0,
    oxygen_concentration: 50.0,
    inspiratory_flow: 30.0,
    exhaled_co2: 40.0,
    humidifier_temperature: 37.0,
    compressor_pressure: 4.0,
    valve_response: 50.0,
    battery_level: 85.0,
    power_consumption: 2.5,
    alarm_frequency: 2.0,
    system_runtime: 1500.0,
    vibration: 0.5,
    internal_temperature: 35.0,
    filter_status: 75.0
  })
  
  const [prediction, setPrediction] = useState<VentilatorPrediction | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  const handleInputChange = (field: keyof VentilatorSensorReading, value: number) => {
    setSensorReading(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const makeVentilatorPrediction = async (): Promise<VentilatorPrediction> => {
    try {
      // Call the Python backend API
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          device_type: 'Ventilator',
          airway_pressure: sensorReading.airway_pressure,
          tidal_volume: sensorReading.tidal_volume,
          respiratory_rate: sensorReading.respiratory_rate,
          oxygen_concentration: sensorReading.oxygen_concentration,
          inspiratory_flow: sensorReading.inspiratory_flow,
          exhaled_co2: sensorReading.exhaled_co2,
          humidifier_temperature: sensorReading.humidifier_temperature,
          compressor_pressure: sensorReading.compressor_pressure,
          valve_response: sensorReading.valve_response,
          battery_level: sensorReading.battery_level,
          power_consumption: sensorReading.power_consumption,
          alarm_frequency: sensorReading.alarm_frequency,
          system_runtime: sensorReading.system_runtime,
          vibration: sensorReading.vibration,
          internal_temperature: sensorReading.internal_temperature,
          filter_status: sensorReading.filter_status
        })
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`)
      }

      const result = await response.json()
      
      if (result.error) {
        throw new Error(result.error)
      }

      return result
    } catch (error: any) {
      console.error('Prediction error:', error)
      // Fallback to simulation if API is not available
      const mockPrediction = simulateVentilatorPrediction()
      throw new Error('API unavailable, using simulation mode. ' + error.message)
    }
  }

  const simulateVentilatorPrediction = (): VentilatorPrediction => {
    // Simple rule-based simulation (fallback)
    let condition = 'Good'
    let failure = 'No'
    let cause = 'Normal Operation'
    let solution = 'Continue monitoring'
    let confidence = 85.0

    if (sensorReading.filter_status < 30 || sensorReading.battery_level < 20) {
      condition = 'Critical'
      failure = 'Yes'
      cause = sensorReading.filter_status < 30 ? 'Clogged Air Filter' : 'Battery Failure Risk'
      solution = sensorReading.filter_status < 30 ? 'Replace filter immediately' : 'Replace or recharge backup battery'
      confidence = 95.0
    } else if (sensorReading.airway_pressure > 45 || sensorReading.internal_temperature > 60) {
      condition = 'Warning'
      failure = 'Yes'
      cause = sensorReading.airway_pressure > 45 ? 'Airway Obstruction' : 'Overheating'
      solution = sensorReading.airway_pressure > 45 ? 'Check tubing and valves' : 'Check cooling system'
      confidence = 80.0
    } else if (sensorReading.system_runtime > 8000) {
      condition = 'Average'
      confidence = 75.0
    }
    
    return {
      condition,
      failure,
      cause,
      solution,
      time_to_failure: Math.random() * 200 + 100,
      confidence,
      probabilities: `Good: ${Math.random() * 30 + 40}%\nAverage: ${Math.random() * 20 + 20}%\nWarning: ${Math.random() * 10 + 10}%\nCritical: ${Math.random() * 5 + 5}%`
    }
  }

  const handleAnalyze = async () => {
    setIsAnalyzing(true)
    try {
      const result = await makeVentilatorPrediction()
      setPrediction(result)
    } catch (error: any) {
      console.error('Analysis failed:', error)
      // Fallback to simulation
      const mockResult = simulateVentilatorPrediction()
      setPrediction(mockResult)
      alert('API unavailable, using simulation mode. ' + error.message)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleSubmit = async () => {
    if (!prediction || !deviceId || !user) return
    
    setIsSaving(true)
    try {
      // First, create maintenance log entry
      const { data: maintenanceLog, error: logError } = await supabase
        .from('maintenance_logs')
        .insert({
          device_id: deviceId,
          uploaded_by: user.id,
          date_of_maintenance: new Date().toISOString(),
          notes: `Ventilator Maintenance Log - AI Analysis: ${prediction.condition} condition, TTF: ${prediction.time_to_failure} days`
        })
        .select()
        .single()

      if (logError) throw logError

      const maintenanceLogId = maintenanceLog.id

      // Save prediction data to ventilator_predictions table
      const { data: predictionResult, error: predictionError } = await supabase
        .from('ventilator_predictions')
        .insert({
          device_id: deviceId,
          maintenance_log_id: maintenanceLogId,
          uploaded_by: user.id,
          airway_pressure: sensorReading.airway_pressure,
          tidal_volume: sensorReading.tidal_volume,
          respiratory_rate: sensorReading.respiratory_rate,
          oxygen_concentration: sensorReading.oxygen_concentration,
          inspiratory_flow: sensorReading.inspiratory_flow,
          exhaled_co2: sensorReading.exhaled_co2,
          humidifier_temperature: sensorReading.humidifier_temperature,
          compressor_pressure: sensorReading.compressor_pressure,
          valve_response: sensorReading.valve_response,
          battery_level: sensorReading.battery_level,
          power_consumption: sensorReading.power_consumption,
          alarm_frequency: sensorReading.alarm_frequency,
          system_runtime: sensorReading.system_runtime,
          vibration: sensorReading.vibration,
          internal_temperature: sensorReading.internal_temperature,
          filter_status: sensorReading.filter_status,
          predicted_condition: prediction.condition,
          predicted_failure: prediction.failure,
          predicted_cause: prediction.cause,
          recommended_solution: prediction.solution,
          time_to_failure_days: prediction.time_to_failure,
          confidence_score: prediction.confidence,
          class_probabilities: prediction.probabilities
        })
        .select()

      if (predictionError) {
        console.error('Error saving prediction:', predictionError)
        throw new Error(`Failed to save prediction: ${predictionError.message}`)
      }

      console.log('Ventilator prediction saved successfully:', predictionResult)

      // Update device status based on AI prediction
      const deviceStatus = getDeviceStatusFromPrediction(prediction.condition)
      const nextMaintenanceDate = calculateNextMaintenanceDate(prediction.time_to_failure)

      const { error: updateError } = await supabase
        .from('devices')
        .update({
          status: deviceStatus,
          last_maintenance_date: new Date().toISOString(),
          next_maintenance_date: nextMaintenanceDate.toISOString()
        })
        .eq('id', deviceId)

      if (updateError) {
        console.error('Error updating device status:', updateError)
      }

      navigate('/dashboard')
      
    } catch (error) {
      console.error('Error saving Ventilator maintenance log:', error)
      alert('Failed to save Ventilator maintenance log. Please try again.')
    } finally {
      setIsSaving(false)
    }
  }

  const getDeviceStatusFromPrediction = (condition: string): 'OK' | 'Warning' | 'Danger' => {
    switch (condition) {
      case 'Good': return 'OK'
      case 'Average': return 'OK'
      case 'Warning': return 'Warning'
      case 'Critical': return 'Danger'
      default: return 'OK'
    }
  }

  const calculateNextMaintenanceDate = (timeToFailure: number): Date => {
    const nextMaintenance = new Date()
    nextMaintenance.setDate(nextMaintenance.getDate() + Math.min(timeToFailure, 90)) // Cap at 90 days
    return nextMaintenance
  }

  const sensorFields = [
    { key: 'airway_pressure', label: 'Airway Pressure', unit: 'cmH2O', icon: Gauge, range: [10, 60] },
    { key: 'tidal_volume', label: 'Tidal Volume', unit: 'ml', icon: Wind, range: [200, 800] },
    { key: 'respiratory_rate', label: 'Respiratory Rate', unit: 'bpm', icon: Activity, range: [8, 30] },
    { key: 'oxygen_concentration', label: 'Oxygen Concentration', unit: '%', icon: Wind, range: [21, 100] },
    { key: 'inspiratory_flow', label: 'Inspiratory Flow', unit: 'L/min', icon: Wind, range: [10, 60] },
    { key: 'exhaled_co2', label: 'Exhaled CO2', unit: 'mmHg', icon: Wind, range: [20, 80] },
    { key: 'humidifier_temperature', label: 'Humidifier Temperature', unit: '°C', icon: Thermometer, range: [30, 45] },
    { key: 'compressor_pressure', label: 'Compressor Pressure', unit: 'bar', icon: Gauge, range: [2, 8] },
    { key: 'valve_response', label: 'Valve Response', unit: 'ms', icon: Activity, range: [20, 200] },
    { key: 'battery_level', label: 'Battery Level', unit: '%', icon: Battery, range: [0, 100] },
    { key: 'power_consumption', label: 'Power Consumption', unit: 'kW', icon: Zap, range: [1, 10] },
    { key: 'alarm_frequency', label: 'Alarm Frequency', unit: 'count', icon: AlertTriangle, range: [0, 50] },
    { key: 'system_runtime', label: 'System Runtime', unit: 'hours', icon: Clock, range: [0, 20000] },
    { key: 'vibration', label: 'Vibration', unit: 'mm/s', icon: Activity, range: [0.1, 5.0] },
    { key: 'internal_temperature', label: 'Internal Temperature', unit: '°C', icon: Thermometer, range: [20, 70] },
    { key: 'filter_status', label: 'Filter Status', unit: '%', icon: Droplets, range: [0, 100] }
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/dashboard')}
            className="flex items-center text-gray-600 hover:text-gray-900 mb-4"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back to Dashboard
          </button>
          
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                <Wind className="w-8 h-8 mr-3 text-green-600" />
                Ventilator Maintenance Log
              </h1>
              <p className="text-gray-600 mt-2">AI-powered failure prediction and maintenance tracking</p>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
              >
                <BarChart3 className="w-5 h-5 mr-2" />
                {isAnalyzing ? 'Analyzing...' : 'Analyze with AI'}
              </button>
              
              {prediction && (
                <button
                  onClick={handleSubmit}
                  disabled={isSaving}
                  className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                >
                  <Save className="w-5 h-5 mr-2" />
                  {isSaving ? 'Saving...' : 'Save Log'}
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Sensor Input */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Ventilator Sensor Readings</h2>
            
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {sensorFields.map((field) => {
                const Icon = field.icon
                return (
                  <div key={field.key} className="flex items-center space-x-4">
                    <Icon className="w-5 h-5 text-gray-400 flex-shrink-0" />
                    <div className="flex-1">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {field.label}
                      </label>
                      <div className="flex items-center space-x-2">
                        <input
                          type="number"
                          value={sensorReading[field.key as keyof VentilatorSensorReading]}
                          onChange={(e) => handleInputChange(field.key as keyof VentilatorSensorReading, parseFloat(e.target.value))}
                          min={field.range[0]}
                          max={field.range[1]}
                          step="0.1"
                          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                        />
                        <span className="text-sm text-gray-500 w-12">{field.unit}</span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* AI Prediction Results */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">AI Analysis Results</h2>
            
            {prediction ? (
              <div className="space-y-6">
                {/* Condition Status */}
                <div className="text-center">
                  <div className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-semibold ${
                    prediction.condition === 'Good' ? 'bg-green-100 text-green-800' :
                    prediction.condition === 'Average' ? 'bg-blue-100 text-blue-800' :
                    prediction.condition === 'Warning' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {prediction.condition === 'Good' ? <CheckCircle className="w-5 h-5 mr-2" /> :
                     prediction.condition === 'Average' ? <CheckCircle className="w-5 h-5 mr-2" /> :
                     prediction.condition === 'Warning' ? <AlertTriangle className="w-5 h-5 mr-2" /> :
                     <AlertTriangle className="w-5 h-5 mr-2" />}
                    {prediction.condition} Condition
                  </div>
                </div>

                {/* Prediction Details */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600">Failure Risk</div>
                    <div className={`text-lg font-semibold ${
                      prediction.failure === 'No' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {prediction.failure}
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600">Confidence</div>
                    <div className="text-lg font-semibold text-blue-600">
                      {prediction.confidence.toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600">Time to Failure</div>
                    <div className="text-lg font-semibold text-orange-600">
                      {prediction.time_to_failure.toFixed(0)} days
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600">Root Cause</div>
                    <div className="text-sm font-medium text-gray-900">
                      {prediction.cause}
                    </div>
                  </div>
                </div>

                {/* Recommended Solution */}
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">Recommended Solution</div>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                    <p className="text-sm text-green-800">{prediction.solution}</p>
                  </div>
                </div>

                {/* Probabilities */}
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">Condition Probabilities</div>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <pre className="text-sm text-gray-700 whitespace-pre-wrap">{prediction.probabilities}</pre>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Wind className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">Click "Analyze with AI" to get Ventilator failure prediction</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default VentilatorMaintenanceLog
