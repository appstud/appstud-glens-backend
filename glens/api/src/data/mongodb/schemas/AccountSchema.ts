import * as mongoose from 'mongoose'
import { DomainAccessRole } from '../../../domain/models/processing/nested/DomainAccessRole'

// tslint:disable-next-line:no-any
export interface AccountDocument extends mongoose.Document {
    id: string
    email: string
    password: string
    authority: DomainAccessRole
    secret: string
    createdDate: Date
}

export const AccountSchema = new mongoose.Schema({
    email: { type: String, required: true },
    password: { type: String },
    authority: { type: String },
    secret: { type: String },
    createdDate: { type: Date },
})

export const DBAccount = mongoose.model('DBAccount', AccountSchema)
