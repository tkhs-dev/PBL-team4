package ac.osaka_u.ics.pbl.data.dao

import org.jetbrains.exposed.dao.id.IntIdTable

object Clients : IntIdTable("clients") {
    val secret = varchar("secret", 255).uniqueIndex()
    val user = varchar("user", 255)
}