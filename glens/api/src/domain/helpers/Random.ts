import * as crypto from 'crypto'
import * as uuid from 'uuid'

export class Random {
    static string(length: number): string {
        return crypto
            .randomBytes(length)
            .toString('hex')
            .substr(0, length)
    }

    static uuid(): string {
        return uuid.v4()
    }
}
