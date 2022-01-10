import { DomainAccessRole } from '../models/processing/nested/DomainAccessRole'

export interface IAccessTokenProvider {
    /**
     * Create a token for a role
     * @param role AccessRole for token
     * @param validity Validity      seconds
     * @param refresh Is this token used for refresh ?
     */
    create(
        role: DomainAccessRole,
        validity: number,
        refresh: boolean
    ): Promise<string>
    read(token: string, verify: boolean): Promise<DomainAccessRole>
    isRole(
        token: string,
        role: DomainAccessRole,
        verify: boolean
    ): Promise<boolean>
    isRefresh(token: string, verify: boolean): Promise<boolean>
}
