import * as IORedis from 'ioredis'
import { Environment } from '../../environment'

export class Redis extends IORedis {
    constructor(environment: Environment) {
        super({
            host: environment.getConfigurationOrDefault(
                'REDIS_HOST',
                'localhost'
            ),
            port: environment.getConfigurationOrDefault('REDIS_PORT', 6379),
        })
    }
}
