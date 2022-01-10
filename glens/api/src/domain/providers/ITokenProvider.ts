import JWTToken from '../models/auth/JWTToken'
import Authentication from '../models/auth/Authentication'
import { DomainAccessRole } from '../models/processing/nested/DomainAccessRole'

export default interface ITokenProvider {
    generateTokens(
        accountId: string,
        secret: string,
        authority: DomainAccessRole
    ): Promise<JWTToken>

    unpackToken(accessToken: string): Promise<Authentication>

    verifyToken(accessToken: string, secret: string): Promise<void>
}
