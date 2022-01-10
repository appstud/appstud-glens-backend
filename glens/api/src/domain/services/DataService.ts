import { IPeopleStorage } from '../../data/IPeopleStorage'
import { forClass } from '../helpers/Logging'
let logger = forClass('DataService')
export class DataService {
    private storage: IPeopleStorage

    constructor(storage: IPeopleStorage) {
        this.storage = storage
    }
    getSummaryInfoOfPeopleBetweenDates(from: Date, to: Date) {
        /*
        process.hrtime
        let hrstartJS=process.hrtime()
        this.storage.getSummaryInfoOfPeopleBetweenDatesJS(from=from,to=to)
        let hrendJS = process.hrtime(hrstartJS)

        let hrstart=process.hrtime()
        this.storage.getSummaryInfoOfPeopleBetweenDates(from=from,to=to)
        let hrend = process.hrtime(hrstart)

        console.info('Execution time JS (hr): %ds %dms', hrendJS[0], hrendJS[1] / 1000000)
        console.info('Execution time (hr): %ds %dms', hrend[0], hrend[1] / 1000000)
        */

        return this.storage.getSummaryInfoOfPeopleBetweenDates(
            (from = from),
            (to = to)
        )
    }
    getDataBetweenDates(from: Date, to: Date) {
        return this.storage.getPeopleDataBetweenDates((from = from), (to = to))
    }

    getPositionInfoOfAPerson(from: Date, to: Date,person_id: string){
        return this.storage.getPositionInfoOfAPerson(from, to, person_id)
    }
}
