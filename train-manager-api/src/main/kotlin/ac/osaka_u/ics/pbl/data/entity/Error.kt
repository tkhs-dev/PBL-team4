package ac.osaka_u.ics.pbl.data.entity

import ac.osaka_u.ics.pbl.data.dao.Errors
import org.jetbrains.exposed.dao.IntEntity
import org.jetbrains.exposed.dao.IntEntityClass
import org.jetbrains.exposed.dao.id.EntityID

class ErrorEntity(id: EntityID<Int>): IntEntity(id){
    companion object : IntEntityClass<ErrorEntity>(Errors)

    var task by Errors.task
    var stackTrace by Errors.stackTrace
    var timestamp by Errors.timestamp
    var client by Errors.client
    var assignment by Errors.assignment
    var clientVersion by Errors.clientVersion
}