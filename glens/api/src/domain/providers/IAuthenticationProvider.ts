import AuthRequest from '../models/auth/AuthRequest'
import { AuthType } from '../models/auth/AuthType'

export default interface IAuthProvider {
    verifyLogin(request: AuthRequest): Promise<boolean>

    supports(type: AuthType): boolean
}
