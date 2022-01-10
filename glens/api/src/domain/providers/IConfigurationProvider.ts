import { DomainConfiguration } from '../models/processing/DomainConfiguration'

export interface IConfigurationProvider {
    getAllConfigurations<T>(): Promise<Array<DomainConfiguration<T>>>
    findConfiguration<T>(name: string): Promise<DomainConfiguration<T> | null>

    getOrCreate<T>(
        document: DomainConfiguration<T>
    ): Promise<DomainConfiguration<T>>
    upsert<T>(document: DomainConfiguration<T>): Promise<DomainConfiguration<T>>
    /**
     * Listen for configuration change and call a function before configuration change
     * @param name Configuration name
     * @param callback Function to call before saving
     * @returns function to stop listening
     */
    onBeforeSave(name: string, callback: CallableFunction): CallableFunction
}
