package ac.osaka_u.ics.pbl.data.dao

import org.jetbrains.exposed.dao.id.IntIdTable
import org.jetbrains.exposed.sql.kotlin.datetime.timestamp

object Errors: IntIdTable("errors") {
    val task = reference("task_id", Tasks.id)
    val stackTrace = text("stack_trace")
    val timestamp = timestamp("timestamp")
    val client = reference("client_id", Clients.id)
    val assignment = reference("assignment_id", Assignments.id)
    val clientVersion = varchar("client_version", 128)
}