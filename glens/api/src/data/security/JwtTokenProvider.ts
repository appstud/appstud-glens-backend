import { decode, sign, TokenExpiredError, verify } from 'jsonwebtoken'
import ITokenProvider from '../../domain/providers/ITokenProvider'
import JWTToken from '../../domain/models/auth/JWTToken'
import Authentication from '../../domain/models/auth/Authentication'
import { Environment } from '../../environment'
import { DomainAccessRole } from '../../domain/models/processing/nested/DomainAccessRole'

export default class JwtTokenProvider implements ITokenProvider {
    private environment: Environment

    constructor(environment: Environment) {
        this.environment = environment
    }

    async generateTokens(
        accountId: string,
        secret: string,
        authority: DomainAccessRole
    ): Promise<JWTToken> {
        const expirationAccess = 86400 // 1 day
        const expirationRefresh = 2592000 // 30 days
        const accessToken = sign(
            { authority: DomainAccessRole },
            this.environment.getConfigurationOrDefault(
                'JWT_SECRET',
                'someverysuperlongtokensomeverysuperlongtokensomeverysuperlongtokensomeverysuperlongtoken'
            ) + secret,
            { subject: accountId.toString(), expiresIn: expirationAccess }
        )
        const refreshToken = sign(
            { authority: 'USER', accessToken: accessToken },
            this.environment.getConfigurationOrDefault(
                'JWT_SECRET',
                'someverysuperlongtokensomeverysuperlongtokensomeverysuperlongtokensomeverysuperlongtoken'
            ) + secret,
            { subject: accountId.toString(), expiresIn: expirationRefresh }
        )
        const today = new Date()
        return new JWTToken(
            accessToken,
            new Date(today.getTime() + 1000 * expirationAccess),
            refreshToken,
            new Date(today.getTime() + 1000 * expirationRefresh)
        )
    }

    async unpackToken(token: string): Promise<Authentication> {
        const token2 = decode(token)
        if (!token2) {
            throw Error('WrongToken')
        }
        if (typeof token2 === 'object') {
            return new Authentication(
                token2.authority,
                token2.iat,
                token2.exp,
                token2.sub,
                'accessToken' in token2
            )
        }
        throw Error('WrongToken')
    }

    async verifyToken(accessToken: string, secret: string): Promise<void> {
        try {
            verify(
                accessToken,
                this.environment.getConfigurationOrDefault(
                    'JWT_SECRET',
                    'someverysuperlongtokensomeverysuperlongtokensomeverysuperlongtokensomeverysuperlongtoken'
                ) + secret
            )
            return
        } catch (err) {
            if (err instanceof TokenExpiredError) {
                throw Error('ExpiredToken')
            }
        }
        throw Error('WrongToken')
    }
}
