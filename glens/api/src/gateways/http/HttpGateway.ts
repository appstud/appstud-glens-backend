import * as fastify from 'fastify'
import { FastifyInstance } from 'fastify'
import { Environment } from '../../environment'
import { IModule } from '../../dependencies'
import { forClass } from '../../domain/helpers/Logging'
const cors = require('fastify-cors')

export class HttpGateway implements IModule {
    private instance: FastifyInstance
    private environment: Environment

    constructor(environment: Environment) {
        this.environment = environment
        this.instance = fastify({
            logger: environment.isTest ? false : forClass('HttpGateway'),
        })

        if (this.environment.getConfigurationOrDefault('HTTP_CORS', false))
            this.instance.register(cors, {
                origin: this.environment.getConfigurationOrDefault(
                    'HTTP_CORS_ORIGIN',
                    '*'
                ),
                methods: this.environment.getConfigurationOrDefault(
                    'HTTP_CORS_METHODS',
                    '*'
                ),
            })
    }

    get router() {
        return this.instance
    }

    get port() {
        return +this.environment.getConfigurationOrDefault('PORT', 8080)
    }

    async start() {
        await this.instance.listen(this.port, '0.0.0.0')
    }

    async stop() {
        await this.instance.close()
    }
}
