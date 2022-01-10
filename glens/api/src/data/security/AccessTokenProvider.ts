import { IAccessTokenProvider } from '../../domain/providers/IAccessTokenProvider'
import { DomainAccessRole } from '../../domain/models/processing/nested/DomainAccessRole'
import * as jwt from 'jsonwebtoken'
import { Token } from './models/Token'
import { Random } from '../../domain/helpers/Random'

export class AccessTokenProvider implements IAccessTokenProvider {
    private readonly salt: string

    constructor() {
        // Will be used to encrypt secret + force to disconnect admin on reload
        this.salt = Random.string(20)
    }

    create(
        role: DomainAccessRole,
        expiresIn: number,
        refresh = false
    ): Promise<string> {
        return new Promise((ok, ko) => {
            jwt.sign(
                { role, refresh },
                this.salt,
                { expiresIn },
                (err, token) => {
                    if (err) return ko(err)
                    return ok(token)
                }
            )
        })
    }

    _decode(token: string): Promise<Token> {
        return new Promise((ok, ko) => {
            ok(jwt.decode(token) as Token)
        })
    }

    _verify(token: string): Promise<Token> {
        return new Promise((ok, ko) => {
            jwt.verify(token, this.salt, (err, payload) => {
                if (err) return ko(err)
                return ok(payload as Token)
            })
        })
    }

    async read(token: string, verify: boolean): Promise<DomainAccessRole> {
        if (verify) return (await this._verify(token)).role

        return (await this._decode(token)).role
    }

    async isRole(
        token: string,
        role: DomainAccessRole,
        verify: boolean
    ): Promise<boolean> {
        const payload = verify
            ? await this._verify(token)
            : await this._decode(token)
        return payload.role === role
    }

    async isRefresh(token: string, verify: boolean): Promise<boolean> {
        try {
            const payload = verify
                ? await this._verify(token)
                : await this._decode(token)
            return payload.refresh
        } catch (e) {
            return false
        }
    }
}
