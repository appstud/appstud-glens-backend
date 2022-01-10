import { Logger as PinoLogger } from 'pino'
import * as pino from 'pino'
import { environment } from '../../environment'

let Logger: PinoLogger
const logger = () => {
    if (!Logger)
        Logger = pino({
            redact: {
                paths: ['email', 'phone', 'password', 'secret'],
                censor: '**REDACTED**',
            },
            level: 'info',
            prettyPrint: environment.isDevelopment,
        })
    return Logger
}

export const forClass = (module: string): PinoLogger => {
    return logger().child({ name: module })
}
