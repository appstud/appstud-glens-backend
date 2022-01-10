import { DomainConfiguration } from '../models/processing/DomainConfiguration'
import { IConfigurationProvider } from '../providers/IConfigurationProvider'

export class ConfigService {
    private provider: IConfigurationProvider

    static configurations = {
        CGU_CONFIGURATION: 'DOMAIN_CGU_CONFIGURATION',
    }

    constructor(provider: IConfigurationProvider) {
        this.provider = provider
    }

    // tslint:disable-next-line:no-any
    async getAllConfigurations(): Promise<Array<DomainConfiguration<any>>> {
        return (await this.provider.getAllConfigurations()).map(
            (configuration: DomainConfiguration<unknown>) => {
                if (configuration.secret) configuration.value = ''
                return configuration
            }
        )
    }

    async addConfiguration<T>(
        name: string,
        value: T
    ): Promise<DomainConfiguration<T>> {
        const configuration = await this.provider.findConfiguration<T>(name)
        return this.provider.upsert(
            new DomainConfiguration(
                name,
                value,
                configuration ? configuration.secret : false
            )
        )
    }

    async getConfiguration<T>(
        name: string,
        protect = true
    ): Promise<DomainConfiguration<T> | null> {
        const configuration = await this.provider.findConfiguration<T>(name)
        if (!configuration) return null
        if (protect && configuration.secret)
            (configuration as DomainConfiguration<unknown>).value = ''
        return configuration
    }
}
