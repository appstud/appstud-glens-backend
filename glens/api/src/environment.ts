import { Logger as PinoLogger } from 'pino'
import * as dotenv from 'dotenv'
import { forClass } from './domain/helpers/Logging'

const DEFAULT_ENVIRONMENT = 'development'
const ENVIRONMENTS = ['test', 'production', 'staging', 'development']

export class Environment {
    private _logger?: PinoLogger
    private _name = ''
    private iniatialized = false

    constructor(name: string) {
        this.name = name
    }

    static get environment() {
        return process.env.NODE_ENV || DEFAULT_ENVIRONMENT
    }

    get logger(): PinoLogger {
        if (!this._logger) this._logger = forClass('Environment')
        return this._logger as PinoLogger
    }

    set name(val) {
        if (!ENVIRONMENTS.includes(val)) {
            this.logger.warn(
                `Unrecognized ${val} environment name. Available: [${ENVIRONMENTS.join(
                    ','
                )}]. Default to 'local' environment`
            )
            this._name = DEFAULT_ENVIRONMENT
            return
        } else this._name = val
    }

    get name() {
        return this._name
    }

    get path() {
        return `${__dirname}/../.env.` + this.name
    }

    async init(environment?: string): Promise<Environment> {
        if (this.iniatialized) return this
        this.iniatialized = true

        if (environment && environment !== this.name) this.name = environment
        await dotenv.config({ path: this.path })
        this.logger.info(
            `Initializing with environment ${this.name} (${this.path})`
        )

        return this
    }

    get isDevelopment() {
        return this.name !== 'production'
    }

    get isTest() {
        return this.name === 'test'
    }

    getConfigurationOrDefault<T>(name: string, defaultValue: T): T {
        return process.env[name]
            ? ((process.env[name] as unknown) as T)
            : defaultValue
    }

    getConfigurationOrThrow(
        name: string,
        message = ' Please provide one to run the application.'
    ) {
        if (!process.env[name])
            throw new Error(
                `Unable to find environment variable ${name}.${message}`
            )

        return process.env[name]
    }
}

export const environment = new Environment(Environment.environment)
