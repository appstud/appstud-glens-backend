import Data from "../domain/models/processing/Data";

export interface IPeopleStorage {
    getPeopleDataBetweenDates(from: Date, to: Date)
    getSummaryInfoOfPeopleBetweenDates(from: Date, to: Date)
    getSummaryInfoOfPeopleBetweenDatesJS(from: Date, to: Date)
    getPositionInfoOfAPerson(from: Date, to: Date, person_id: string)
    savePeopleData(message: string)
    saveData(data: Data[])
}
