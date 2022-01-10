import { Workbook, Row, Cell, Column, Worksheet } from 'exceljs'
import { forClass } from '../../domain/helpers/Logging'
import { IPeopleStorage } from '../IPeopleStorage'
import Data from "../../domain/models/processing/Data";
let logger = forClass('Excel')

export class ExcelDatabase implements IPeopleStorage {
    private pathOfExcelFile: string
    private wb: Workbook
    private ws: Worksheet

    constructor(path: string) {
        this.pathOfExcelFile = path
        this.wb = new Workbook()
        this.ws =
            this.wb.getWorksheet('My Sheet') || this.wb.addWorksheet('My Sheet')
        this.initializeWorkSheet()
        this.wb.xlsx
            .readFile(this.pathOfExcelFile)
            .then(ok => logger.info('Excel File already exists'))
            .catch(err =>
                this.wb.xlsx
                    .writeFile(this.pathOfExcelFile)
                    .then(() => logger.info('Created excel file'))
            )
    }


    initializeWorkSheet() {
        let columns = [
            { header: 'id', key: 'id', width: 10 },
            { header: 'age', key: 'age', width: 32 },
            { header: 'gender', key: 'gender', width: 10, outlineLevel: 1 },
            { header: 'date', key: 'date', width: 10, outlineLevel: 1 },
            {
                header: 'hairColor',
                key: 'hairColor',
                width: 10,
                outlineLevel: 1,
            },
        ] as Column[]
        this.ws.columns = columns
    }

    async getPositionInfoOfAPerson(from: Date, to: Date, person_id: string){
        //TODO
        return {}
    }

    async getSummaryInfoOfPeopleBetweenDatesJS(from: Date, to: Date) {
        //TODO
        return {}
    }

    async getSummaryInfoOfPeopleBetweenDates(from: Date, to: Date) {
        /*      let filteringParam=0.95;
            this.ws.eachRow( (row, rowNumber)=> {

                if(row.getCell('id').value?.toString()===id_.toString()){
                    found=true
                    row.getCell('age').value=(((filteringParam)*(row.getCell('age').value||parsed_JSON[id_].age||null)+(1-filteringParam)*(parsed_JSON[id_].age || row.getCell('age').value ||null)).toFixed(2) as any) as number
                    //row.getCell('age').value= (row.getCell('age').value!="NA" && parsed_JSON[id_].age && filteringParameter*parsed_JSON[id_].age+(1-filteringParameter)*(row.getCell('age').value as number))||parsed_JSON[id_].age ||"NA"
                    row.getCell('gender').value= parsed_JSON[id_].sex||"NA"
                    row.getCell('hairColor').value= parsed_JSON[id_].hairColor||"NA"
                    row.getCell('date').value=new Date()
                    row.commit();
                }

            });
            */
        //TODO later
        this.getPeopleDataBetweenDates(from, to)
    }
    async getPeopleDataBetweenDates(from: Date, to: Date) {
        let answer = {}
        let id_
        let event_count = 0
        this.wb.xlsx.readFile(this.pathOfExcelFile)
        this.ws =
            this.wb.getWorksheet('My Sheet') || this.wb.addWorksheet('My Sheet')
        this.initializeWorkSheet()

        this.ws.eachRow((row, rowNumber) => {
            if (
                (row.getCell('date').value as Date) >= from &&
                (row.getCell('date').value as Date) <= to
            ) {
                id_ = row.getCell('id').value.toString()
                answer[event_count++] = {}
                for (let i = 0; i < this.ws.columns.length; i++) {
                    if (this.ws.columns[i].key) {
                        answer[event_count - 1][
                            this.ws.columns[i].key
                        ] = row.getCell(this.ws.columns[i].key).value
                    }
                }
            }
        })

        return answer
    }

    savePeopleData(message: string) {
        this.wb.xlsx
            .readFile(this.pathOfExcelFile)
            .then(() => {
                this.ws =
                    this.wb.getWorksheet('My Sheet') ||
                    this.wb.addWorksheet('My Sheet')
                this.initializeWorkSheet()
                let parsed_JSON = JSON.parse(message).data

                for (let id_ in parsed_JSON) {
                    let found = false
                    if (isNaN(parseFloat(id_))) {
                        //not recognized/ no recognition/ do not store the values
                        continue
                    }

                    this.ws.addRow({
                        id: id_,
                        age: parsed_JSON[id_].age || null,
                        gender: parsed_JSON[id_].sex || null,
                        date: new Date(),
                        hairColor: parsed_JSON[id_].hairColor || null,
                    })
                }

                this.wb.xlsx
                    .writeFile(this.pathOfExcelFile)
                    .then(() => logger.info('success'))
                    .catch(err => logger.info(err))
            })
            .catch(err => logger.info(err))
    }

    saveData(data: Data[]) {
        throw Error("Not implemented on this provider")
    }
}
