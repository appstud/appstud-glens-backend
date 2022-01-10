import { DomainConfiguration } from '../../../domain/models/processing/DomainConfiguration'
import * as mongoose from 'mongoose'
import { Schema } from 'mongoose'

// tslint:disable-next-line:no-any
export interface ConfigurationDocument
    extends mongoose.Document,
        DomainConfiguration<unknown> {
    id: string
}

export const ConfigurationSchema = new mongoose.Schema({
    name: { type: String, required: true },
    value: Schema.Types.Mixed,
    secret: { type: Boolean },
})

export const DBConfiguration = mongoose.model<ConfigurationDocument>(
    'Configuration',
    ConfigurationSchema
)
