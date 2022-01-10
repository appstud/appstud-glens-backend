import * as mongoose from 'mongoose'

export interface DataDocument extends mongoose.Document {
    id: string
    person_id: string
    age: number
    gender: string
    hairColor: string
    date: Date
    mask: boolean
    glasses: boolean
    pose_data: [Number]
}

export const DataSchema = new mongoose.Schema({
    person_id: { type: String },
    age: { type: Number },
    gender: { type: String },
    hairColor: { type: String },
    date: { type: Date },
    mask: { type: Boolean},
    glasses: { type: Boolean},
    pose_data: { type: [Number] },
})

export const DBData = mongoose.model('DBData', DataSchema)
