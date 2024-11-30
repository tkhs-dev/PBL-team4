package ac.osaka_u.ics.pbl

// APIの例外を実装.基本的にはHTTPステータスコードに対応する例外を実装する.
sealed class ApiException: Exception() {
    class NotFoundException: ApiException()
    class BadRequestException(override val message: String?): ApiException()
    class UnauthorizedException: ApiException()
    class ForbiddenException: ApiException()
    open class InternalServerErrorException(override val message: String?): ApiException()

    // 内部エラーを詳細に分割
    class DatabaseException(private val detail: String?): InternalServerErrorException("[Database error] $detail")
}