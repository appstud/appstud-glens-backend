import { DomainConfiguration } from '../../../domain/models/processing/DomainConfiguration'
import { IConfigurationProvider } from '../../../domain/providers/IConfigurationProvider'
import {
    DBConfiguration,
    ConfigurationDocument,
} from '../schemas/ConfigurationSchema'

export class ConfigurationProvider implements IConfigurationProvider {
    private onBeforeSaveHooks: { [k: string]: Function[] } = {}

    private static toConfig<T>(
        document: ConfigurationDocument
    ): DomainConfiguration<T> {
        return new DomainConfiguration<T>(
            document.name,
            document.value as T,
            document.secret
        )
    }

    private async _createConfiguration<T>(
        name: string,
        value: T,
        secret: boolean
    ): Promise<ConfigurationDocument> {
        const newConfig = new DBConfiguration()
        newConfig.name = name
        newConfig.value = value
        newConfig.secret = secret
        if (this.onBeforeSaveHooks[name])
            await Promise.all(
                this.onBeforeSaveHooks[name].map(hook => hook(newConfig))
            )

        return newConfig.save()
    }

    private async _getConfiguration(
        name: string
    ): Promise<ConfigurationDocument | null> {
        return DBConfiguration.findOne({ name }).exec()
    }

    private async _updateConfiguration<T>(
        document: ConfigurationDocument,
        update: DomainConfiguration<T>
    ): Promise<ConfigurationDocument> {
        document.value = update.value
        document.secret = update.secret
        if (this.onBeforeSaveHooks[document.name])
            await Promise.all(
                this.onBeforeSaveHooks[document.name].map(hook =>
                    hook(document)
                )
            )

        return document.save()
    }

    private async _getOrCreate<T>(
        name: string,
        defaultValue: T,
        secret = false
    ): Promise<ConfigurationDocument> {
        return (
            (await this._getConfiguration(name)) ||
            this._createConfiguration(name, defaultValue, secret)
        )
    }

    // tslint:disable-next-line:no-any
    async getAllConfigurations(): Promise<Array<DomainConfiguration<any>>> {
        const configs = await DBConfiguration.find({}).exec()
        return configs.map(result => ConfigurationProvider.toConfig(result))
    }

    async findConfiguration<T>(
        name: string
    ): Promise<DomainConfiguration<T> | null> {
        const config = await this._getConfiguration(name)
        if (!config) return null

        return ConfigurationProvider.toConfig<T>(config)
    }

    async getOrCreate<T>(
        document: DomainConfiguration<T>
    ): Promise<DomainConfiguration<T>> {
        return ConfigurationProvider.toConfig<T>(
            await this._getOrCreate(
                document.name,
                document.value,
                document.secret
            )
        )
    }

    async upsert<T>(
        document: DomainConfiguration<T>
    ): Promise<DomainConfiguration<T>> {
        const dbConfig = await this._getConfiguration(document.name)
        if (dbConfig)
            return ConfigurationProvider.toConfig<T>(
                await this._updateConfiguration(dbConfig, document)
            )
        return ConfigurationProvider.toConfig<T>(
            await this._createConfiguration(
                document.name,
                document.value,
                document.secret
            )
        )
    }

    onBeforeSave(name: string, callback: CallableFunction): CallableFunction {
        this.onBeforeSaveHooks[name] = this.onBeforeSaveHooks[name] || []
        this.onBeforeSaveHooks[name].push(callback)
        return () => {
            this.onBeforeSaveHooks[name] = this.onBeforeSaveHooks[name].filter(
                fn => fn !== callback
            )
        }
    }
}
